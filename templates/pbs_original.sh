#!/bin/bash

#PBS -S /bin/bash
#PBS -N {{job.name}}
#PBS -A {{user.sponsor}}
#PBS -M {{user.email}}
{% for resource, limit in job.resources.items() -%}
    #PBS -l {{resource|e}}={{limit|e}}
{% endfor -%}
#PBS -q {{job.queue}}
#PBS -o {{job.stdout}}
#PBS -e {{job.stderr}}
#PBS -m {{job.emails}}

# set singularity environment variables
{% for var, value in job.environment.items() -%}
    {{var}}={{value}}
{% endfor %}

# write metadata to log files
#>&2 echo 'job_id: ' {% raw -%}$PBS_JOBID{%- endraw %}

# setup job resources, e.g. load modules
{{job.setup}}

# execute job
{{job.cmd}}
