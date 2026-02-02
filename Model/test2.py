import dash
from dash import dcc, html, Input, Output, State, ALL, ctx
import dash_bootstrap_components as dbc
from datetime import datetime
import json

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Priority colors mapping
PRIORITY_COLORS = {
    1: '#dc3545',  # Red - Urgent
    2: '#fd7e14',  # Orange - High
    3: '#ffc107',  # Yellow - Medium
    4: '#28a745'   # Green - Low
}

PRIORITY_LABELS = {
    1: 'Urgent',
    2: 'High',
    3: 'Medium',
    4: 'Low'
}

# Sample initial tasks
initial_tasks = [
    {
        'id': 1,
        'title': 'Complete project proposal',
        'description': 'Draft and finalize the Q1 project proposal document',
        'priority': 1,
        'completed': False,
        'subtasks': [
            {'id': 11, 'title': 'Research requirements', 'completed': True},
            {'id': 12, 'title': 'Write executive summary', 'completed': False},
            {'id': 13, 'title': 'Create budget breakdown', 'completed': False}
        ]
    },
    {
        'id': 2,
        'title': 'Review team feedback',
        'description': 'Go through all feedback from last sprint review',
        'priority': 2,
        'completed': False,
        'subtasks': [
            {'id': 21, 'title': 'Compile feedback notes', 'completed': False},
            {'id': 22, 'title': 'Schedule follow-up meeting', 'completed': False}
        ]
    },
    {
        'id': 3,
        'title': 'Update documentation',
        'description': 'Update API documentation with new endpoints',
        'priority': 3,
        'completed': True,
        'subtasks': []
    },
    {
        'id': 4,
        'title': 'Organize desk workspace',
        'description': 'Clean and reorganize desk for better productivity',
        'priority': 4,
        'completed': False,
        'subtasks': [
            {'id': 41, 'title': 'Clear unnecessary items', 'completed': False}
        ]
    }
]

def create_task_card(task):
    """Create a card component for a single task"""
    priority = task['priority']
    task_id = task['id']
    
    # Create subtask items
    subtask_items = []
    for subtask in task.get('subtasks', []):
        subtask_items.append(
            dbc.FormCheck(
                id={'type': 'subtask-checkbox', 'task_id': task_id, 'subtask_id': subtask['id']},
                type='checkbox',
                label=subtask['title'],
                checked=subtask['completed'],
                className='ms-4 mb-1',
                style={'fontSize': '0.9rem'}
            )
        )
    
    card = dbc.Card(
        [
            dbc.CardBody(
                [
                    html.Div(
                        [
                            # Checkbox and title
                            html.Div(
                                [
                                    dbc.FormCheck(
                                        id={'type': 'task-checkbox', 'index': task_id},
                                        type='checkbox',
                                        checked=task['completed'],
                                        className='me-3',
                                        style={'transform': 'scale(1.2)'}
                                    ),
                                    html.H5(
                                        task['title'],
                                        className='mb-0',
                                        style={
                                            'textDecoration': 'line-through' if task['completed'] else 'none',
                                            'color': '#6c757d' if task['completed'] else '#212529'
                                        }
                                    )
                                ],
                                style={'display': 'flex', 'alignItems': 'center'}
                            ),
                            # Priority badge
                            dbc.Badge(
                                PRIORITY_LABELS[priority],
                                color='light',
                                className='ms-auto',
                                style={
                                    'backgroundColor': PRIORITY_COLORS[priority],
                                    'color': 'white',
                                    'fontWeight': 'bold'
                                }
                            )
                        ],
                        style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}
                    ),
                    # Description
                    html.P(
                        task['description'],
                        className='mt-2 mb-2 text-muted',
                        style={'fontSize': '0.95rem'}
                    ),
                    # Subtasks
                    html.Div(
                        subtask_items,
                        className='mt-2'
                    ) if subtask_items else None
                ]
            )
        ],
        className='mb-3 shadow-sm',
        style={'borderLeft': f'4px solid {PRIORITY_COLORS[priority]}'}
    )
    
    return card

# App layout
app.layout = dbc.Container(
    [
        html.H1('📝 Todo List', className='mt-4 mb-4 text-center'),
        
        # Add new task section
        dbc.Card(
            dbc.CardBody(
                [
                    html.H5('Add New Task', className='mb-3'),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Input(id='new-task-title', placeholder='Task title...', type='text'),
                                width=12, lg=6
                            ),
                            dbc.Col(
                                dbc.Select(
                                    id='new-task-priority',
                                    options=[
                                        {'label': f'{PRIORITY_LABELS[i]} Priority', 'value': i}
                                        for i in range(1, 5)
                                    ],
                                    value=3
                                ),
                                width=12, lg=3, className='mt-2 mt-lg-0'
                            ),
                            dbc.Col(
                                dbc.Button('Add Task', id='add-task-btn', color='primary', className='w-100'),
                                width=12, lg=3, className='mt-2 mt-lg-0'
                            )
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Textarea(
                                    id='new-task-description',
                                    placeholder='Task description (optional)...',
                                    style={'height': '60px'}
                                ),
                                width=12, className='mt-2'
                            )
                        ]
                    )
                ]
            ),
            className='mb-4 shadow-sm'
        ),
        
        # Task list
        html.Div(id='task-list'),
        
        # Store for tasks data
        dcc.Store(id='tasks-store', data=initial_tasks)
    ],
    fluid=True,
    style={'maxWidth': '900px'}
)

@app.callback(
    Output('tasks-store', 'data'),
    [
        Input('add-task-btn', 'n_clicks'),
        Input({'type': 'task-checkbox', 'index': ALL}, 'checked'),
        Input({'type': 'subtask-checkbox', 'task_id': ALL, 'subtask_id': ALL}, 'checked')
    ],
    [
        State('new-task-title', 'value'),
        State('new-task-description', 'value'),
        State('new-task-priority', 'value'),
        State('tasks-store', 'data')
    ]
)
def update_tasks(n_clicks, task_checks, subtask_checks, new_title, new_desc, new_priority, tasks):
    """Update tasks based on user interactions"""
    if not ctx.triggered:
        return tasks
    
    triggered_id = ctx.triggered[0]['prop_id']
    
    # Add new task
    if 'add-task-btn' in triggered_id and new_title:
        new_id = max([t['id'] for t in tasks], default=0) + 1
        new_task = {
            'id': new_id,
            'title': new_title,
            'description': new_desc or '',
            'priority': int(new_priority),
            'completed': False,
            'subtasks': []
        }
        tasks.append(new_task)
        return tasks
    
    # Update task checkbox
    if 'task-checkbox' in triggered_id:
        triggered_dict = ctx.triggered_id
        task_id = triggered_dict['index']
        for i, task in enumerate(tasks):
            if task['id'] == task_id:
                tasks[i]['completed'] = task_checks[
                    [t['id'] for t in tasks].index(task_id)
                ]
                break
        return tasks
    
    # Update subtask checkbox
    if 'subtask-checkbox' in triggered_id:
        triggered_dict = ctx.triggered_id
        task_id = triggered_dict['task_id']
        subtask_id = triggered_dict['subtask_id']
        
        for task in tasks:
            if task['id'] == task_id:
                for i, subtask in enumerate(task['subtasks']):
                    if subtask['id'] == subtask_id:
                        # Find the correct index in the flattened subtask_checks list
                        check_index = 0
                        found = False
                        for t in tasks:
                            for st in t.get('subtasks', []):
                                if t['id'] == task_id and st['id'] == subtask_id:
                                    found = True
                                    break
                                check_index += 1
                            if found:
                                break
                        
                        if check_index < len(subtask_checks):
                            subtask['completed'] = subtask_checks[check_index]
                        break
                break
        return tasks
    
    return tasks

@app.callback(
    [
        Output('task-list', 'children'),
        Output('new-task-title', 'value'),
        Output('new-task-description', 'value')
    ],
    Input('tasks-store', 'data')
)
def render_tasks(tasks):
    """Render the task list sorted by priority"""
    if not tasks:
        return [html.P('No tasks yet. Add one above!', className='text-muted text-center mt-4')], '', ''
    
    # Sort tasks by priority (1 = highest priority)
    sorted_tasks = sorted(tasks, key=lambda x: (x['priority'], x['id']))
    
    task_cards = [create_task_card(task) for task in sorted_tasks]
    
    return task_cards, '', ''

if __name__ == '__main__':
    app.run(debug=True, port=8050)
