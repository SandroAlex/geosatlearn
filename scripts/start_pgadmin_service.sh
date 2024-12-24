#!/bin/bash

# Create servers.json file from environmental variables.
cat <<EOF > /pgadmin4/servers.json
{
  "Servers": {
    "1": {
      "Name": "${DATABASE_NAME}",
      "Group": "Servers",
      "Host": "${DATABASE_HOST}",
      "Port": ${DATABASE_PORT},
      "MaintenanceDB": "postgres",
      "Username": "${DATABASE_USER}",
      "Password": "${DATABASE_PASSWORD}",
      "SSLMode": "prefer",
      "ConnectNow": true
    }
  }
}
EOF

# Run the original entrypoint script.
exec /entrypoint.sh "$@"