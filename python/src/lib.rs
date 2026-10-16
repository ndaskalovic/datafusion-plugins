//! datafusion-python bindings: PyCapsule glue around the core crate.
//!
//! Objects exported to datafusion-python cross a shared-library boundary
//! (`datafusion-ffi`). Each side has its own copy of tokio, so the plans
//! built here run on this library's own runtime ([`runtime`]), which they
//! enter themselves on `execute` and on every poll.

use std::ptr::NonNull;
use std::sync::{Arc, OnceLock};

use arrow::array::{RecordBatch, StructArray};
use arrow::datatypes::{Schema, SchemaRef};
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema, from_ffi};
use datafusion::common::{DataFusionError, Result as DfResult};
use datafusion_ffi::proto::logical_extension_codec::FFI_LogicalExtensionCodec;
use datafusion_ffi::table_provider::FFI_TableProvider;
use datafusion_plugins::{
    BatchReader, JoinAsof, JoinAsofMarkouts, Markout, PartitionSource, PlanBytesTable, PlanNode,
    PreparedSource, StreamInputTable, last_plan,
};
use pyo3::exceptions::{PyImportError, PyStopIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict, PyTuple};

/// This library's tokio runtime. Worker threads: `DATAFUSION_PLUGINS_THREADS`
/// if set, else the number of CPUs. Read once, at first use.
fn runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        let threads = std::env::var("DATAFUSION_PLUGINS_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|n| *n > 0)
            .unwrap_or_else(|| {
                std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
            });
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(threads)
            .thread_name("datafusion-plugins")
            .enable_all()
            .build()
            .expect("failed to start the datafusion-plugins tokio runtime")
    })
}

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
        PyValueError::new_err(
            "expected a datafusion SessionContext or a datafusion_logical_extension_codec capsule",
        )
    })?;
    let ptr: NonNull<FFI_LogicalExtensionCodec> = capsule
        .pointer_checked(Some(c"datafusion_logical_extension_codec"))?
        .cast();
    let codec = unsafe { ptr.as_ref() };
    let found = unsafe { (codec.version)() };
    if found != datafusion_ffi::version() {
        return Err(PyImportError::new_err(format!(
            "datafusion-plugins was built for DataFusion {} but the installed datafusion \
             package is {found}.x; install datafusion=={}.*",
            datafusion_ffi::version(),
            datafusion_ffi::version()
        )));
    }
    Ok(codec.clone())
}

// ---------------------------------------------------------------------------
// Nodes: one builder per node, from the Python keyword arguments.
// ---------------------------------------------------------------------------

fn arg<'py, T: FromPyObjectOwned<'py>>(args: &Bound<'py, PyDict>, key: &str) -> PyResult<T> {
    match args.get_item(key)? {
        Some(v) => v.extract().map_err(Into::into),
        None => Err(PyValueError::new_err(format!("missing argument {key:?}"))),
    }
}

fn join_asof(args: &Bound<'_, PyDict>) -> PyResult<Arc<dyn PlanNode>> {
    Ok(Arc::new(
        JoinAsof::new_with(arg::<String>(args, "left_on")?, arg::<String>(args, "right_on")?)
            .by_left_right(arg::<Vec<String>>(args, "by_left")?, arg::<Vec<String>>(args, "by_right")?)
            .allow_exact_matches(arg(args, "allow_exact_matches")?)
            .suffix(arg::<String>(args, "suffix")?)
            .assume_sorted(arg(args, "assume_sorted")?)
            .check_sortedness(arg(args, "check_sortedness")?),
    ))
}

fn join_asof_markouts(args: &Bound<'_, PyDict>) -> PyResult<Arc<dyn PlanNode>> {
    let markouts = arg::<Vec<String>>(args, "markouts")?
        .iter()
        .map(|m| Markout::parse(m))
        .collect::<datafusion::common::Result<Vec<_>>>()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(Arc::new(
        JoinAsofMarkouts::new(arg::<String>(args, "on")?, markouts)
            .by_left_right(arg::<Vec<String>>(args, "by_left")?, arg::<Vec<String>>(args, "by_right")?)
            .allow_exact_matches(arg(args, "allow_exact_matches")?)
            .assume_sorted(arg(args, "assume_sorted")?)
            .check_sortedness(arg(args, "check_sortedness")?),
    ))
}

/// The node registry: name -> builder.
fn make_node(name: &str, args: &Bound<'_, PyDict>) -> PyResult<Arc<dyn PlanNode>> {
    match name {
        "join_asof" => join_asof(args),
        "join_asof_markouts" => join_asof_markouts(args),
        other => Err(PyValueError::new_err(format!("unknown node {other:?}"))),
    }
}

fn provider_capsule<'py>(
    py: Python<'py>,
    session: &Bound<'py, PyAny>,
    table: Arc<dyn datafusion::catalog::TableProvider>,
) -> PyResult<Bound<'py, PyCapsule>> {
    let codec = logical_codec(session)?;
    let ffi = FFI_TableProvider::new_with_ffi_codec(table, false, Some(runtime().handle().clone()), codec);
    PyCapsule::new_with_value(py, ffi, c"datafusion_table_provider")
}

fn df_err(e: PyErr) -> DataFusionError {
    DataFusionError::External(Box::new(e))
}

/// A pyarrow schema (anything with `__arrow_c_schema__`) as an Arrow schema.
fn import_schema(obj: &Bound<'_, PyAny>) -> PyResult<SchemaRef> {
    let cap = obj.call_method0("__arrow_c_schema__")?;
    let cap = cap.cast::<PyCapsule>()?;
    let ptr: NonNull<FFI_ArrowSchema> = cap.pointer_checked(Some(c"arrow_schema"))?.cast();
    let schema = Schema::try_from(unsafe { ptr.as_ref() })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(Arc::new(schema))
}

/// A batch (anything with `__arrow_c_array__`) as a RecordBatch, moving the
/// data out of the capsule (zero copy).
fn import_batch(obj: &Bound<'_, PyAny>, schema: &SchemaRef) -> PyResult<RecordBatch> {
    // datafusion-python's internal RecordBatch exports through pyarrow.
    let obj = if obj.hasattr("to_pyarrow")? { obj.call_method0("to_pyarrow")? } else { obj.clone() };
    let pair = obj.call_method0("__arrow_c_array__")?;
    let (sc, ac): (Bound<'_, PyAny>, Bound<'_, PyAny>) = pair.extract()?;
    let (sc, ac) = (sc.cast::<PyCapsule>()?, ac.cast::<PyCapsule>()?);
    let sp: NonNull<FFI_ArrowSchema> = sc.pointer_checked(Some(c"arrow_schema"))?.cast();
    let ap: NonNull<FFI_ArrowArray> = ac.pointer_checked(Some(c"arrow_array"))?.cast();
    let array = unsafe { FFI_ArrowArray::from_raw(ap.as_ptr()) };
    let data = unsafe { from_ffi(array, sp.as_ref()) }.map_err(|e| PyValueError::new_err(e.to_string()))?;
    let batch = RecordBatch::from(StructArray::from(data));
    Ok(batch.clone().with_schema(Arc::clone(schema)).unwrap_or(batch))
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

/// One host partition stream, read on a plugin blocking thread. Each `next`
/// takes the GIL briefly; datafusion-python releases it while it waits on
/// its own runtime, so partitions are read concurrently.
struct PyReader {
    stream: Py<PyAny>,
    schema: SchemaRef,
}

impl Iterator for PyReader {
    type Item = DfResult<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        Python::attach(|py| match self.stream.bind(py).call_method0("next") {
            Ok(b) => Some(import_batch(&b, &self.schema).map_err(df_err)),
            Err(e) if e.is_instance_of::<PyStopIteration>(py) => None,
            Err(e) => Some(Err(df_err(e))),
        })
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
                df.call_method1("select", PyTuple::new(py, columns)?)?
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
            let streams = self.df.bind(py).call_method0("execute_stream_partitioned")?;
            let mut out: Vec<BatchReader> = Vec::new();
            for s in streams.try_iter()? {
                out.push(Box::new(PyReader { stream: s?.unbind(), schema: Arc::clone(&self.schema) }));
            }
            Ok(out)
        })
        .map_err(df_err)
    }
}

/// A node over DataFrames the host executes (any input), as a table provider.
#[pyclass(name = "StreamTable", module = "datafusion_plugins._internal", frozen)]
struct PyStreamTable {
    table: Arc<StreamInputTable>,
}

#[pymethods]
impl PyStreamTable {
    #[new]
    fn new(node: &str, inputs: Vec<Bound<'_, PyAny>>, args: Bound<'_, PyDict>) -> PyResult<Self> {
        let node = make_node(node, &args)?;
        let sources = inputs
            .iter()
            .map(|df| {
                let schema = import_schema(&df.call_method0("schema")?)?;
                Ok(Arc::new(PyDataFrameSource { df: df.clone().unbind(), schema }) as Arc<dyn PartitionSource>)
            })
            .collect::<PyResult<Vec<_>>>()?;
        let table = StreamInputTable::try_new(sources, node, Some(runtime().handle().clone()))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { table: Arc::new(table) })
    }

    fn __datafusion_table_provider__<'py>(
        &self,
        py: Python<'py>,
        session: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        provider_capsule(py, &session, Arc::clone(&self.table) as _)
    }
}

/// A node over serialized logical plans, as a table provider.
#[pyclass(name = "PlanTable", module = "datafusion_plugins._internal", frozen)]
struct PyPlanTable {
    table: Arc<PlanBytesTable>,
}

#[pymethods]
impl PyPlanTable {
    #[new]
    fn new(py: Python<'_>, node: &str, inputs: Vec<Vec<u8>>, args: Bound<'_, PyDict>) -> PyResult<Self> {
        let node = make_node(node, &args)?;
        let handle = runtime().handle().clone();
        let table = py
            .detach(|| PlanBytesTable::try_new(inputs, node, Some(handle)))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { table: Arc::new(table) })
    }

    fn __datafusion_table_provider__<'py>(
        &self,
        py: Python<'py>,
        session: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        provider_capsule(py, &session, Arc::clone(&self.table) as _)
    }
}

/// The physical plan most recently built by a node in this process.
#[pyfunction(name = "last_plan")]
fn py_last_plan() -> Option<String> {
    last_plan()
}

/// Limit the memory of the plugin's plans (`None`: unbounded). `pool` is
/// "fair" (FairSpillPool) or "greedy" (GreedyMemoryPool).
#[pyfunction]
#[pyo3(signature = (limit, pool = "fair"))]
fn set_memory_limit(limit: Option<usize>, pool: &str) -> PyResult<()> {
    use datafusion::execution::memory_pool::{FairSpillPool, GreedyMemoryPool, MemoryPool};
    let pool: Option<Arc<dyn MemoryPool>> = match (limit, pool) {
        (None, _) => None,
        (Some(l), "fair") => Some(Arc::new(FairSpillPool::new(l))),
        (Some(l), "greedy") => Some(Arc::new(GreedyMemoryPool::new(l))),
        (_, other) => return Err(PyValueError::new_err(format!("pool must be 'fair' or 'greedy', got {other:?}"))),
    };
    datafusion_plugins::set_memory_pool(pool);
    Ok(())
}

/// The major DataFusion (FFI) version this library was built against.
#[pyfunction]
fn ffi_version() -> u64 {
    datafusion_ffi::version()
}

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPlanTable>()?;
    m.add_class::<PyStreamTable>()?;
    m.add_function(wrap_pyfunction!(py_last_plan, m)?)?;
    m.add_function(wrap_pyfunction!(ffi_version, m)?)?;
    m.add_function(wrap_pyfunction!(set_memory_limit, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
