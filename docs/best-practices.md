# Best Practices

## Be disciplined about Tasks

Avoid the temptation to add additional parameters or variables to your
functions directly. It is worth the effort to encapsulate all input
variables into `Task` objects.

## Maintain backwards compatibility in Task definitions

If you need to add a new option (parameter) to a Task, give it a default
value and make sure the `.fn` property is backwards compatible. This means
that you can append additional parts to the string if the new parameter
is anything other than its default value, but if the new parameter
is set to its default avoid adding it to `fn`. By maintaining backwards
compatibility, you can aggregate datasets taken before and after the change.
However, too many changes can make the Task hard to follow so try to think
of all the parameters up front. If you need significant changes, it might
be a new Task.

## Write driver scripts in Python

Python driver scripts can be concise and highly readable while also
avoiding any ambiguity over what data will be collected and in which order.
If you're writing command-line parsing and/or doing for loops in bash,
something has gone wrong. Configuration options live in the Python
driver script. Consequently, you may have to edit some "code" before launching
a job. The overhead and inflexibility of introducing configuration files
is likely not worth it.

## Separate library functionality out of Tasks

There may be a tendency to develop more and more logic into task functions.
Avoid this temptation! Factor out 'business logic' into well-designed
library functions. Task definitions and driver scripts should merely translate
configuration options into calls to library functionality and handle I/O.

## Save everything

Be exhaustive in metadata pertaining to your experiment. You should likely
be saving job ids, calibration ids, durations of various parts of your
task, circuits, and more. By default, a timestamp is saved with each
file.

