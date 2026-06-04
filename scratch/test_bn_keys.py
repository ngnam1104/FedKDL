import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tasks.detection_2d.models.yolo_wrapper import StudentModel

student = StudentModel(ckpt='yolo12n_warmup.pt', rank=4, nc=4, use_lora=True)
keys = student.trainable_state_dict().keys()

running_keys = [k for k in keys if 'running' in k or 'tracked' in k]
print(f"Number of running/tracked keys in payload: {len(running_keys)}")
if len(running_keys) > 0:
    print(running_keys[:5])
