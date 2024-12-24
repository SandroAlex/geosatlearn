# Project name.
PROJECT_NAME="geosatlearn-project"

# Environment used in building API: development (true) or production (false).
DEVELOPMENT=true

# Select development or production docker compose file.
ifeq ($(DEVELOPMENT),true)
	COMPOSE_FILE="docker-compose.dev.yml"
else
	# Not implemented yet.
	COMPOSE_FILE="docker-compose.yml"
endif

# BUILDING
#####################################################################
# Build local microservices.
docker-build-all:
	docker-compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} up --build --detach ;

# Turn on all local micro services.
docker-up-all:
	docker-compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} up --detach ;

# Stop all local micro services.
docker-down-all:
	docker-compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} down --volumes ;
#####################################################################

# DEBUG AND TESTS
#####################################################################
# Show a panoramic view of containers.
docker-config:
	docker-compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} config ;

# Show logs for all micro services.
docker-logs:
	docker-compose --project-name ${PROJECT_NAME} logs --follow --timestamps ;

# Debug web running container.
docker-debug-web-service:
	docker-compose --project-name ${PROJECT_NAME} exec web-service sh -c "/bin/bash" ;

# Run jupyterlab on web service.
docker-exec-jupyterlab:
	docker-compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} exec worker-service sh -c "jupyter-lab --allow-root --ip 0.0.0.0" ;
#####################################################################