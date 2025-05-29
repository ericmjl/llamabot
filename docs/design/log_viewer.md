# LlamaBot Log Viewer Design Document

## Overview
The LlamaBot Log Viewer is a web-based interface for inspecting and analyzing LlamaBot call logs. It provides a comprehensive view of all interactions with the bot, including prompts, responses, and tool usage.

## Core Features

### 1. Log Inspection
- **Split Panel Layout**
  - Left panel: List of log entries
  - Right panel: Detailed view of selected log
  - Both panels scroll independently
  - Fixed headers for easy navigation

- **Log Entry List (Left Panel)**
  - Sortable columns:
    - ID
    - Object Name
    - Timestamp
    - Model Name
    - Temperature
    - Prompts Used
  - Clickable rows to view details
  - Visual indicators for helpful/not helpful ratings
  - Compact view with essential information

- **Log Details (Right Panel)**
  - Full conversation history
  - Expandable/collapsible messages
  - Syntax highlighting for code blocks
  - Tool call visualization
  - Prompt template display
  - Rating controls (helpful/not helpful)

### 2. Filtering and Search
- **Text Search**
  - Real-time filtering as you type
  - Search across all fields
  - Highlight matching text

- **Function Name Filter**
  - Dropdown of available prompt functions
  - Shows version count for each function
  - Multi-select capability

### 3. Export Functionality
- **Export Formats**
  - OpenAI format (JSONL)
  - Support for additional formats in future

- **Export Options**
  - Filter by text search
  - Filter by function name
  - Export only helpful responses
  - Include/exclude specific fields

### 4. Prompt Comparison
- **Version Comparison**
  - Side-by-side diff view
  - Syntax highlighting
  - Line-by-line comparison
  - Highlight changes

- **Selection Interface**
  - Dropdown to select prompt versions
  - Preview of selected prompts
  - Easy switching between versions

### 5. Experiment View
- **Experiment List**
  - Name and run count
  - Timestamp of last run
  - Success/failure indicators

- **Run Details**
  - Metrics visualization
  - Message log links
  - Prompt versions used
  - Performance statistics

## Technical Implementation

### Data Structures
- Maintain existing database schema
- No changes to current data models
- Leverage existing relationships between tables

### UI Components
- Use HTMX for dynamic updates
- Bootstrap for responsive layout
- Custom CSS for specialized components
- Jinja2 templates for server-side rendering

### Templates Structure
- **Base Template** (`base.html`)
  - Common layout structure
  - Navigation bar
  - Footer
  - Common CSS/JS includes
  - HTMX setup

- **Log Viewer Templates**
  - `index.html`: Main container with tab navigation
  - `log_table.html`: Left panel log list
  - `log_details.html`: Right panel log details
  - `log_tbody.html`: Dynamic log table body for HTMX updates
  - `message_log.html`: Message list component
  - `rating_buttons.html`: Helpful/not helpful rating controls

- **Prompt Comparison Templates**
  - `prompt_compare.html`: Main comparison view
  - `prompt_selector.html`: Version selection interface
  - `prompt_diff.html`: Side-by-side diff view

- **Experiment Templates**
  - `experiment_list.html`: List of experiments
  - `experiment_details.html`: Run details and metrics
  - `experiment_metrics.html`: Metrics visualization

- **Macros** (`macros.html`)
  - `message_expansion`: Message display with expand/collapse
  - `prompt_display`: Prompt template rendering
  - `tool_call`: Tool call visualization
  - `metric_card`: Metric display component
  - `diff_view`: Diff visualization

### Router Structure
- **Log Router** (`/logs`)
  - `GET /`: Main log viewer page
  - `GET /filtered_logs`: HTMX endpoint for filtered log list
  - `GET /{log_id}`: Individual log details
  - `GET /{log_id}/expand`: Expand all messages
  - `GET /{log_id}/collapse`: Collapse all messages
  - `POST /{log_id}/rate`: Rate log as helpful/not helpful
  - `GET /export/{format}`: Export logs in specified format

- **Prompt Router** (`/prompts`)
  - `GET /history`: View prompt version history
  - `GET /functions`: Get list of prompt functions
  - `GET /{prompt_hash}`: View specific prompt version
  - `GET /compare`: Compare two prompt versions

- **Experiment Router** (`/experiments`)
  - `GET /list`: List all experiments
  - `GET /details`: View experiment details
  - `GET /runs`: List runs for an experiment
  - `GET /metrics`: Get experiment metrics

- **Common Patterns**
  - All endpoints use FastAPI dependency injection for DB sessions
  - Consistent error handling with HTTPException
  - HTMX integration for dynamic updates
  - Proper type hints and docstrings
  - Logging for debugging and monitoring

### Static Files Structure
- **CSS** (`static/styles.css`)
  - Base styles for layout and components
  - Responsive design utilities
  - Custom component styles
  - Dark/light theme support
  - Print styles for exports

- **JavaScript** (`static/script.js`)
  - HTMX event handlers
  - Modal management
  - Export functionality
  - Keyboard shortcuts
  - Dynamic UI updates

- **Third-party Dependencies**
  - Core Dependencies (loaded from CDN with SRI)
    - Bootstrap 5.3.6 (CSS only)
    - HTMX 2.0.0
    - Highlight.js 11.11.1

  - Loading Strategy
    - Defer non-critical CSS/JS
    - Use async loading for highlight.js
    - Implement resource hints (preconnect)
    - Bundle and minify custom CSS/JS
    - Use SRI hashes for security

  - Version Management
    - Lock dependency versions
    - Regular security audits
    - Automated dependency updates
    - Fallback CDN sources

- **Asset Organization**
  - Modular CSS with component-specific styles
  - Vanilla JS for minimal dependencies
  - No build step required
  - Easy to maintain and extend

### Performance Considerations
- Pagination for large log sets
- Lazy loading of log details
- Efficient database queries
- Caching of frequently accessed data

## User Experience

### Navigation
- Clear visual hierarchy
- Consistent layout across views
- Intuitive filtering controls
- Quick access to common actions

### Responsiveness
- Mobile-friendly design
- Adaptive layout for different screen sizes
- Touch-friendly controls
- Keyboard shortcuts for power users

### Visual Feedback
- Loading indicators
- Success/error messages
- Clear status indicators
- Helpful tooltips

## Security
- Input sanitization
- SQL injection prevention
- XSS protection
- Rate limiting

## Accessibility
- ARIA labels
- Keyboard navigation
- Screen reader support
- High contrast mode
