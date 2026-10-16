//! datafusion-python bindings: PyCapsule glue around the core crate.
//!
//! Objects exported to datafusion-python cross a shared-library boundary
//! (`datafusion-ffi`). Each side has its own copy of tokio, so the plans
//! built here run on this library's own runtime ([`runtime`]), which they
//! enter themselves on `execute` and on every poll.

use std::ptr::NonNull;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use arrow::array::{RecordBatch, RecordBatchOptions, StructArray};
use arrow::compute::cast;
use arrow::datatypes::{Schema, SchemaRef};
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema, from_ffi};
use datafusion::common::{DataFusionError, Result as DfResult, exec_err};
use datafusion::execution::memory_pool::{FairSpillPool, GreedyMemoryPool, MemoryPool};
use datafusion_ffi::proto::logical_extension_codec::FFI_LogicalExtensionCodec;
use datafusion_ffi::table_provider::FFI_TableProvider;
use datafusion_plugins::{
    BatchReader, HostInputs, JoinAsof, Markout, NodeInputs, NodeTable, PartitionSource, PlanInputs, PreparedSource,
};
use pyo3::create_exception;
use pyo3::exceptions::{PyImportError, PyStopIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict, PyTuple};

create_exception!(_internal, PlanRouteUnavailable, PyValueError, "The inputs can't be planned by the plugin.");

/// This library's tokio runtime (worker threads: `TOKIO_WORKER_THREADS`, or
/// one per CPU).
fn runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .thread_name("datafusion-plugins")
            .enable_all()
            .build()
            .expect("failed to start the datafusion-plugins tokio runtime")
    })
}

/// The memory pool of node plans planned from now on (`set_memory_limit`).
static MEMORY_POOL: Mutex<Option<Arc<dyn MemoryPool>>> = Mutex::new(None);

/// The host session's logical extension codec. `session` is a
/// `SessionContext` (the internal one, or the Python wrapper) or a bare
/// `datafusion_logical_extension_codec` capsule; it is duck-typed, as
/// datafusion-python asks.
fn logical_codec(session: &Bound<'_, PyAny>) -> PyResult<FFI_LogicalExtensionCodec> {
    let obj = if session.hasattr("__datafusion_logical_extension_codec__")? {
        session.call_method1("__datafusion_logical_extension_codec__", (session,))?
    } else {
        session.clone()
    };
    let capsule = obj.cast::<PyCapsule>().map_err(|_| {
        PyValueError::new_err("expected a datafusion SessionContext or a datafusion_logical_extension_codec capsule")
    })?;
    let ptr: NonNull<FFI_LogicalExtensionCodec> =
        capsule.pointer_checked(Some(c"datafusion_logical_extension_codec"))?.cast();
    // SAFETY: the capsule name guarantees the pointee type, and the capsule
    // (alive for this call) owns it.
    let codec = unsafe { ptr.as_ref() };
    // SAFETY: `version` is a plain function pointer set by the codec's library.
    let found = unsafe { (codec.version)() };
    if found != datafusion_ffi::version() {
        return Err(PyImportError::new_err(format!(
            "datafusion-plugins was built for DataFusion {0} but the installed datafusion package is {found}.x; \
             install datafusion=={0}.*",
            datafusion_ffi::version()
        )));
    }
    Ok(codec.clone())
}

fn arg<'py, T: FromPyObjectOwned<'py>>(args: &Bound<'py, PyDict>, key: &str) -> PyResult<T> {
    match args.get_item(key)? {
        Some(v) => v.extract().map_err(Into::into),
        None => Err(PyValueError::new_err(format!("missing argument {key:?}"))),
    }
}

/// `join_asof` (with `markouts`: `join_asof_markouts`) from the Python
/// keyword arguments.
fn join_asof(args: &Bound<'_, PyDict>) -> PyResult<JoinAsof> {
    let mut node = JoinAsof::new_with(arg::<String>(args, "left_on")?, arg::<String>(args, "right_on")?)
        .by_left_right(arg::<Vec<String>>(args, "by_left")?, arg::<Vec<String>>(args, "by_right")?)
        .allow_exact_matches(arg(args, "allow_exact_matches")?)
        .assume_sorted(arg(args, "assume_sorted")?)
        .check_sortedness(arg(args, "check_sortedness")?);
    if let Some(suffix) = args.get_item("suffix")? {
        node = node.suffix(suffix.extract::<String>()?);
    }
    if let Some(ms) = args.get_item("markouts")? {
        let ms = ms
            .extract::<Vec<String>>()?
            .iter()
            .map(|m| Markout::parse(m))
            .collect::<DfResult<Vec<_>>>()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        node = node.markouts(ms);
    }
    Ok(node)
}

/// The node registry.
fn make_node(name: &str, args: &Bound<'_, PyDict>) -> PyResult<Arc<dyn datafusion_plugins::PlanNode>> {
    match name {
        "join_asof" => Ok(Arc::new(join_asof(args)?)),
        other => Err(PyValueError::new_err(format!("unknown node {other:?}"))),
    }
}

fn df_err(e: PyErr) -> DataFusionError {
    DataFusionError::External(Box::new(e))
}

/// A pyarrow schema (anything with `__arrow_c_schema__`) as an Arrow schema.
fn import_schema(obj: &Bound<'_, PyAny>) -> PyResult<SchemaRef> {
    let cap = obj.call_method0("__arrow_c_schema__")?;
    let ptr: NonNull<FFI_ArrowSchema> = cap.cast::<PyCapsule>()?.pointer_checked(Some(c"arrow_schema"))?.cast();
    // SAFETY: the capsule name guarantees the pointee type; it is only read.
    let schema = Schema::try_from(unsafe { ptr.as_ref() }).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(Arc::new(schema))
}

/// A batch (anything with `__arrow_c_array__`) as a RecordBatch of
/// `schema`, moving the data out of the capsule (zero copy).
fn import_batch(obj: &Bound<'_, PyAny>, schema: &SchemaRef) -> PyResult<RecordBatch> {
    let pair = obj.call_method1("__arrow_c_array__", (obj.py().None(),))?;
    let (sc, ac): (Bound<'_, PyAny>, Bound<'_, PyAny>) = pair.extract()?;
    let sp: NonNull<FFI_ArrowSchema> = sc.cast::<PyCapsule>()?.pointer_checked(Some(c"arrow_schema"))?.cast();
    let ap: NonNull<FFI_ArrowArray> = ac.cast::<PyCapsule>()?.pointer_checked(Some(c"arrow_array"))?.cast();
    // SAFETY: the capsule names guarantee the pointee types. `from_raw`
    // moves the array out (leaving it released, so the capsule's destructor
    // does nothing); the schema is only read.
    let array = unsafe { FFI_ArrowArray::from_raw(ap.as_ptr()) };
    // SAFETY: `array` and the schema describe the same exported array.
    let data = unsafe { from_ffi(array, sp.as_ref()) }.map_err(|e| PyValueError::new_err(e.to_string()))?;
    conform(RecordBatch::from(StructArray::from(data)), schema).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// `batch` as a batch of `schema`: nested field names and nullability may
/// differ (cast), and a NULL-free column may be declared non-nullable; other
/// differences are errors.
fn conform(batch: RecordBatch, schema: &SchemaRef) -> DfResult<RecordBatch> {
    if batch.schema().fields() == schema.fields() {
        return Ok(batch.with_schema(Arc::clone(schema))?);
    }
    if batch.num_columns() != schema.fields().len() {
        return exec_err!("host input batch has {} columns, expected {}", batch.num_columns(), schema.fields().len());
    }
    let cols = batch
        .columns()
        .iter()
        .zip(schema.fields())
        .map(|(c, f)| match c.data_type() {
            t if t == f.data_type() => Ok(Arc::clone(c)),
            t if t.equals_datatype(f.data_type()) => Ok(cast(c, f.data_type())?),
            t => exec_err!("host input column '{}' is {t}, expected {}", f.name(), f.data_type()),
        })
        .collect::<DfResult<Vec<_>>>()?;
    let opts = RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
    Ok(RecordBatch::try_new_with_options(Arc::clone(schema), cols, &opts)?)
}

/// A datafusion-python DataFrame as an input the host executes.
#[derive(Debug)]
struct PyDataFrameSource {
    df: Py<PyAny>,
    schema: SchemaRef,
}

#[derive(Debug)]
struct PyPrepared {
    df: Py<PyAny>,
    schema: SchemaRef,
    partitions: usize,
}

/// One host partition stream (datafusion-python's internal stream), read on
/// a plugin blocking thread. Each `next` takes the GIL briefly;
/// datafusion-python releases it while it waits on its own runtime, so
/// partitions are read concurrently.
struct PyReader {
    /// `None` once dropped.
    stream: Option<Py<PyAny>>,
    schema: SchemaRef,
    /// The batches export `__arrow_c_array__` themselves (datafusion-python
    /// 55's don't: they go through pyarrow).
    direct: Option<bool>,
}

impl Iterator for PyReader {
    type Item = DfResult<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        let stream = self.stream.as_ref()?;
        Python::attach(|py| {
            let b = match stream.bind(py).call_method0("__next__") {
                Ok(b) => b,
                Err(e) if e.is_instance_of::<PyStopIteration>(py) => return None,
                Err(e) => return Some(Err(df_err(e))),
            };
            let direct = *self.direct.get_or_insert_with(|| b.hasattr("__arrow_c_array__").unwrap_or(false));
            let b = if direct { Ok(b) } else { b.call_method0("to_pyarrow") };
            Some(b.and_then(|b| import_batch(&b, &self.schema)).map_err(df_err))
        })
    }
}

impl Drop for PyReader {
    fn drop(&mut self) {
        // Release the host stream now (a cancelled query's stream would
        // otherwise live until some thread next takes the GIL).
        if let Some(s) = self.stream.take() {
            let _ = Python::try_attach(|_| drop(s));
        }
    }
}

impl PartitionSource for PyDataFrameSource {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn prepare(&self, columns: &[String]) -> DfResult<Arc<dyn PreparedSource>> {
        Python::attach(|py| -> PyResult<Arc<dyn PreparedSource>> {
            let df = self.df.bind(py);
            let df = if columns.len() == self.schema.fields().len() {
                df.clone()
            } else {
                // Quoted: a bare string would be parsed (case folded, dots
                // split).
                let col = py.import("datafusion")?.getattr("col")?;
                let exprs = columns
                    .iter()
                    .map(|c| col.call1((format!("\"{}\"", c.replace('"', "\"\"")),)))
                    .collect::<PyResult<Vec<_>>>()?;
                df.call_method1("select", PyTuple::new(py, exprs)?)?
            };
            let n = df.call_method0("execution_plan")?.getattr("partition_count")?;
            let n = if n.is_callable() { n.call0()? } else { n };
            let partitions: usize = n.extract()?;
            let schema = import_schema(&df.call_method0("schema")?)?;
            Ok(Arc::new(PyPrepared { df: df.unbind(), schema, partitions }))
        })
        .map_err(df_err)
    }
}

impl PreparedSource for PyPrepared {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn partition_count(&self) -> usize {
        self.partitions
    }

    fn open(&self) -> DfResult<Vec<BatchReader>> {
        Python::attach(|py| -> PyResult<Vec<BatchReader>> {
            // The internal DataFrame's streams: no Python wrapper frames per
            // batch.
            let df = self.df.bind(py);
            let df = if df.hasattr("df")? { df.getattr("df")? } else { df.clone() };
            let mut out: Vec<BatchReader> = Vec::new();
            for s in df.call_method0("execute_stream_partitioned")?.try_iter()? {
                out.push(Box::new(PyReader { stream: Some(s?.unbind()), schema: Arc::clone(&self.schema), direct: None }));
            }
            Ok(out)
        })
        .map_err(df_err)
    }
}

/// A node as a table provider, over serialized plans or DataFrames.
#[pyclass(name = "NodeTable", module = "datafusion_plugins._internal", frozen)]
struct PyNodeTable {
    table: Arc<NodeTable>,
}

impl PyNodeTable {
    fn new(node: &str, inputs: Arc<dyn NodeInputs>, args: &Bound<'_, PyDict>) -> PyResult<Self> {
        let node = make_node(node, args)?;
        let table = NodeTable::try_new(inputs, node)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .with_runtime(runtime().handle().clone())
            .with_memory_pool(|| MEMORY_POOL.lock().unwrap_or_else(PoisonError::into_inner).clone());
        Ok(Self { table: Arc::new(table) })
    }
}

#[pymethods]
impl PyNodeTable {
    /// Node `node` over serialized logical plans, planned by the plugin.
    /// Raises `PlanRouteUnavailable` if they don't decode or don't only read
    /// local files.
    #[staticmethod]
    fn from_plans(py: Python<'_>, node: &str, inputs: Vec<Vec<u8>>, args: Bound<'_, PyDict>) -> PyResult<Self> {
        let inputs = py.detach(|| PlanInputs::try_new(inputs)).map_err(|e| PlanRouteUnavailable::new_err(e.to_string()))?;
        Self::new(node, Arc::new(inputs), &args)
    }

    /// Node `node` over DataFrames datafusion-python executes.
    #[staticmethod]
    fn from_frames(node: &str, inputs: Vec<Bound<'_, PyAny>>, args: Bound<'_, PyDict>) -> PyResult<Self> {
        let sources = inputs
            .iter()
            .map(|df| {
                let schema = import_schema(&df.call_method0("schema")?)?;
                Ok(Arc::new(PyDataFrameSource { df: df.clone().unbind(), schema }) as Arc<dyn PartitionSource>)
            })
            .collect::<PyResult<Vec<_>>>()?;
        Self::new(node, Arc::new(HostInputs::new(sources)), &args)
    }

    /// The physical plan of the latest scan, if planned yet.
    fn last_plan(&self) -> Option<String> {
        self.table.last_plan()
    }

    fn __datafusion_table_provider__<'py>(
        &self,
        py: Python<'py>,
        session: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let codec = logical_codec(&session)?;
        let table = Arc::clone(&self.table) as Arc<dyn datafusion::catalog::TableProvider>;
        let ffi = FFI_TableProvider::new_with_ffi_codec(table, false, Some(runtime().handle().clone()), codec);
        PyCapsule::new_with_value(py, ffi, c"datafusion_table_provider")
    }
}

/// Limit the memory of node queries run from now on (`None`: unbounded).
/// `pool` is "fair" (FairSpillPool) or "greedy" (GreedyMemoryPool); one pool
/// is shared by all those queries.
#[pyfunction]
#[pyo3(signature = (limit, pool = "fair"))]
fn set_memory_limit(limit: Option<usize>, pool: &str) -> PyResult<()> {
    let pool: Option<Arc<dyn MemoryPool>> = match (limit, pool) {
        (None, _) => None,
        (Some(l), "fair") => Some(Arc::new(FairSpillPool::new(l))),
        (Some(l), "greedy") => Some(Arc::new(GreedyMemoryPool::new(l))),
        (_, other) => return Err(PyValueError::new_err(format!("pool must be 'fair' or 'greedy', got {other:?}"))),
    };
    *MEMORY_POOL.lock().unwrap_or_else(PoisonError::into_inner) = pool;
    Ok(())
}

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNodeTable>()?;
    m.add_function(wrap_pyfunction!(set_memory_limit, m)?)?;
    m.add("PlanRouteUnavailable", m.py().get_type::<PlanRouteUnavailable>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
