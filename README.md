# tcc-random-forest

## This script is executed by the container at startup.

> #!/bin/bash
> mysql -D abvdb -u root -p < /docker-entrypoint-initdb.d/database.sql
