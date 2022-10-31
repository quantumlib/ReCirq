FROM python:3.9

WORKDIR /src

RUN python -m pip install --upgrade pip && pip install pytest
COPY . .
RUN python dev_tools/write-ci-requirements.py --relative-cirq-version=current --all-extras
RUN pip install -r ci-requirements.txt && pip install --no-deps -e .
RUN RECIRQ_IMPORT_FAILSAFE=y pytest -v