#!/bin/bash

#PBS -S /bin/bash
#PBS -N {{name}}
#PBS -A {{sponsor}}
#PBS -M {{email}}
{% for resource, limit in resources.items() -%}
#PBS -l {{resource|e}}={{limit|e}}
{% endfor -%}
#PBS -q {{queue}}
#PBS -o {{stdout}}
#PBS -e {{stderr}}
#PBS -m {{emails}}

# activate anaconda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate odelaydev

# set singularity environment variables
{% for var, value in environment.items() -%}
    {{var}}={{value}}
{% endfor %}

# write metadata to log files
#>&2 echo 'job_id: ' {% raw -%}$PBS_JOBID{%- endraw %}

# setup job resources, e.g. load modules
{{setup}}

# execute job
{{cmd}}
