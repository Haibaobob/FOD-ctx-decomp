import os
import time

class scan():
    def __init__(self,
                subject_id,
                session_id = None,
                dataset_name=None,
                modality=None,
                acq_scheme=None):
        self.subject_id = subject_id
        self.session_id = session_id
        self.dataset_name = dataset_name
        self.modality = modality
        self.acq_scheme = acq_scheme
        self.info = f"subjec_id: {subject_id}, session_id: {session_id}"
        self.name = f"{subject_id}_{session_id}"

    @property
    def rel_path(self):
        if self.session_id is None:
            return f"{self.subject_id}"
        else:
            return f"{self.subject_id}/{self.session_id}"
        