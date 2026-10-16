//! datafusion-python bindings: PyCapsule glue around the core crate.
//!
//! Objects exported to datafusion-python cross a shared-library boundary
//! (`datafusion-ffi`). Each side has its own copy of tokio, so the plans
//! built here run on this library's own runtime ([`runtime`]), which they
//! enter themselves on `execute` and on every poll.

use std::ptr::NonNull;
use std::sync::{Arc, OnceLock};

use datafusion_ffi::proto::logical_extension_codec::FFI_LogicalExtensionCodec;
use datafusion_ffi::table_provider::FFI_TableProvider;
use datafusion_plugins::{AsofOptions, AsofPlanTable, JoinAsof, last_plan};
use pyo3::exceptions::{PyImportError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

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

/// A `join_asof` of two serialized logical plans, as a table provider.
#[pyclass(name = "AsofJoinTable", module = "datafusion_plugins._internal", frozen)]
struct PyAsofJoinTable {
    table: Arc<AsofPlanTable>,
}

#[pymethods]
impl PyAsofJoinTable {
    #[new]
    #[pyo3(signature = (left, right, left_on, right_on, by_left, by_right, allow_exact_matches = true, suffix = "_right".to_string(), check_sortedness = true))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        left: Vec<u8>,
        right: Vec<u8>,
        left_on: String,
        right_on: String,
        by_left: Vec<String>,
        by_right: Vec<String>,
        allow_exact_matches: bool,
        suffix: String,
        check_sortedness: bool,
    ) -> PyResult<Self> {
        let join = JoinAsof::new_with(left_on, right_on)
            .by_left_right(by_left, by_right)
            .allow_exact_matches(allow_exact_matches)
            .suffix(suffix)
            .check_sortedness(check_sortedness)
            .options(AsofOptions::from_env());
        let handle = runtime().handle().clone();
        let table = py
            .detach(|| AsofPlanTable::try_new(left, right, join, Some(handle)))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            table: Arc::new(table),
        })
    }

    fn __datafusion_table_provider__<'py>(
        &self,
        py: Python<'py>,
        session: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let codec = logical_codec(&session)?;
        let ffi = FFI_TableProvider::new_with_ffi_codec(
            Arc::clone(&self.table) as _,
            false,
            Some(runtime().handle().clone()),
            codec,
        );
        PyCapsule::new_with_value(py, ffi, c"datafusion_table_provider")
    }
}

/// The physical plan of the most recent asof join planned in this process.
#[pyfunction(name = "last_plan")]
fn py_last_plan() -> Option<String> {
    last_plan()
}

/// The major DataFusion (FFI) version this library was built against.
#[pyfunction]
fn ffi_version() -> u64 {
    datafusion_ffi::version()
}

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAsofJoinTable>()?;
    m.add_function(wrap_pyfunction!(py_last_plan, m)?)?;
    m.add_function(wrap_pyfunction!(ffi_version, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
