#!/bin/bash

nohup python dask_run.py "$@" >/dev/null 2>&1 &