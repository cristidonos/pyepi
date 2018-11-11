"""This script will generate HTML reports:
The RAW_DATA folder needs to be defined and have the following structure:

RAW_DATA
├── subject (i.e. SEEG72)
│   ├── DTI (diffusion dicoms)
│   ├── DWI (this one gets created during preprocessing, contains diffusion dicoms converted to Nifti and bval/bvec files)
│   ├── T1  (T1 dicoms)
│   ├── T2  (T2 dicoms, optional)
│   └── Patient Data.xlsx  (format detailed in io.load_contacts , optional)
│   └── subject.ppr  (ppr file containing the final trajectories, optional)
│   └── SPES.xls  (Excel file with SPES responses, optional)

The output will be generated in Freesurfer's $SUBJECTS_DIR folder.

To be called with 1 positional arguments:
    - SUBJECT_ID
And optional additional arguments to enable/disable various preprocessing steps:
    - implantationscheme / noimplantationscheme (optional, if second argument exist verbose=True)
    - circleplot / nocircleplot
    - verbose (print output)

ex: python3 generate_report SEEG72 verbose implantationscheme circleplot

Email notifications may be sent when the script finishes

"""

import os
import platform
import subprocess
import sys

import nbformat
import psutil
from nbparameterise import extract_parameters, replace_definitions, parameter_values

from pyepi.tools import paths, notifications

RAW_DATA, RAW_DATA_NATIVE, SUBJECTS_DIR, SUBJECTS_DIR_NATIVE = paths.set_paths(platform=platform.node())

MAX_RAM_SIZE = psutil.virtual_memory()[0] / 2. ** 30  # in GB
CPU_COUNT = psutil.cpu_count()

# PARAMETERS (using paths in WSL format, ie. /mnt/d/....)
send_notification_when_done = True
send_notifications_to_email = 'cristidonos@yahoo.com'

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Subject ID needs to be specified as the first argument:\n    generate_report TEST_SUBJECT01 ')
        sys.exit()

    subj = sys.argv[1]
    verbose = False

    email_body = [subj]

    # spes_file = RAW_DATA_NATIVE + os.sep + subj + os.sep + 'SPES.xls'

    # if not os.path.isfile(spes_file):
    #     print("    -> SPES.xls file does not exist in subject's RAWDATA folder.")
    #     sys.exit()

    reports_dir = SUBJECTS_DIR_NATIVE + os.sep + subj + os.sep + 'reports'
    os.makedirs(reports_dir, exist_ok=True)
    report_file = os.path.join(reports_dir, 'report.ipynb')

    with open(os.path.join(sys.path[0], 'spes_report.ipynb')) as f:
        nb = nbformat.read(f, as_version=4)

    # Get a list of Parameter objects
    orig_parameters = extract_parameters(nb)

    params = parameter_values(orig_parameters,
                              subj=subj,
                              subj_dir=SUBJECTS_DIR_NATIVE)
    # Make a notebook object with these definitions, and execute it.
    new_nb = replace_definitions(nb, params, execute=False)
    with open(report_file, 'w') as f:
        nbformat.write(new_nb, f)

    subprocess.run(['jupyter', 'nbconvert', '--execute',
                    '--ExecutePreprocessor.timeout=7200',
                    '--ExecutePreprocessor.kernel_name=python3',
                    '--to=html', report_file])

    if send_notification_when_done:
        try:
            notifications.sendmail(to_address=send_notifications_to_email, subject=subj + ' Report job done.',
                                   message=subj + '\n')
        except:
            pass
