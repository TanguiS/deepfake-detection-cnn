import sys
from io import StringIO

from log_io.logger import Logger

log = Logger()

stdout_original = sys.stdout
sys.stdout = StringIO()

print("WTF")
print("WTF3")

captured_message = sys.stdout.getvalue()

test_msg = captured_message.split('\n')[:-1]

sys.stdout = stdout_original



log.info(captured_message)
print("WTF2")

