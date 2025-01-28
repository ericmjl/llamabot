# Secure Script Execution for LLM Agents

A secure pattern for executing self-written code from LLM agents in local environments.

## Overview

This design document outlines a secure approach for allowing LLM agents to write and execute Python code in a sandboxed environment. The pattern uses Docker containers and PEP 723 metadata to create a secure execution environment for agent-generated code.

## Core Components

### 1. Script Metadata

Scripts are written with PEP 723 metadata headers that specify dependencies and requirements:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests<3",
#   "rich",
# ]
# auth = "agent-id-hash"
# purpose = "task-description"
# timestamp = "iso-timestamp"
# ///

# agent code here
```

### 2. Docker Configuration

The execution environment uses the Astral UV image for optimal Python package management:

```dockerfile
# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install the project into `/app`
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

# Run as non-root user for security
USER nobody

# Run the script with uv
CMD ["uv", "run", "--system-site-packages=false"]
```

### 3. Script Execution Flow

1. Agent generates Python code with metadata
2. Code is written to a temporary directory
3. Docker container is built with security constraints
4. Script is executed in isolated environment
5. Results are captured and returned to agent

### 4. Security Measures

The implementation includes multiple layers of security:

1. Container Restrictions:
   - Read-only root filesystem
   - No network access by default
   - Limited CPU (1 core) and memory (512MB)
   - Mounted script directory is read-only
   - Results directory is write-only
   - All capabilities dropped
   - No privilege escalation

2. Code Validation:
   - PEP 723 metadata verification
   - Execution timeout enforcement
   - Structured error handling
   - Output validation

## Implementation

The core implementation consists of two main classes:

1. `ScriptMetadata`: Pydantic model for script metadata
2. `ScriptExecutor`: Handles script writing and secure execution

Example usage:

```python
@tool
def write_and_execute_script(
    code: str,
    python_version: str = ">=3.11",
    dependencies: Optional[List[str]] = None,
    purpose: str = "",
    timeout: int = 30,
) -> Dict[str, Any]:
    """Write and execute a Python script in a secure sandbox."""
    metadata = ScriptMetadata(
        requires_python=python_version,
        dependencies=dependencies or [],
        auth=str(uuid4()),
        purpose=purpose,
        timestamp=datetime.now(),
    )

    executor = ScriptExecutor()
    script_path = executor.write_script(code, metadata)
    return executor.run_script(script_path, timeout)
```

## Benefits

1. **Security**: Multiple layers of isolation and restrictions
2. **Flexibility**: Agents can write custom code solutions
3. **Dependency Management**: Clean environment for each execution
4. **Resource Control**: Strict limits on compute resources
5. **Auditability**: Metadata tracking and logging

## Testing

The implementation includes comprehensive tests:

- Script writing and metadata handling
- Execution in sandbox environment
- Timeout enforcement
- Error handling
- Dependency management
- Resource restrictions

## Future Enhancements

Potential areas for improvement:

1. Network access controls for specific domains
2. Resource usage monitoring and logging
3. Script validation and static analysis
4. Caching of commonly used dependencies
5. Support for additional runtime environments
