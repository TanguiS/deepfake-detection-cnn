import sys
from io import StringIO
from typing import Dict, Callable


def capture_message_from_monitored_function(
        printer: Callable[[str], None],
        function_to_monitor: Callable,
        monitored_function_kwargs: Dict[str, any]):
    stdout_original = sys.stdout
    sys.stdout = StringIO()
    function_to_monitor(**monitored_function_kwargs)
    captured_message = sys.stdout.getvalue()
    messages = captured_message.split('\n')
    sys.stdout = stdout_original
    if len(messages) == 0:
        return
    for message in messages:
        printer(message)
