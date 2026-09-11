// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay_rust_grpc_worker_plugin_example::DocumentationWorker;
use nemo_relay_worker::{Result, serve_plugin};

#[tokio::main]
async fn main() -> Result<()> {
    #[cfg(windows)]
    {
        // Windows sends console Ctrl+C to Relay and its workers. Let Relay
        // finish its requests and shut this worker down through the gRPC API.
        // SAFETY: A null handler with TRUE sets this process's Ctrl+C ignore
        // flag. No callback or borrowed memory is passed to Windows.
        if unsafe { windows_sys::Win32::System::Console::SetConsoleCtrlHandler(None, 1) } == 0 {
            return Err(nemo_relay_worker::WorkerSdkError::Transport(format!(
                "could not ignore console Ctrl+C: {}",
                std::io::Error::last_os_error()
            )));
        }
    }
    serve_plugin(DocumentationWorker).await
}
