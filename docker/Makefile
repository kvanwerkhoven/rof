
.PHONY: build deploy help

help:
	    @echo "Makefile commands:"
	    @echo "\tbuild - Build the image(s)"
	    @echo "\tdeploy - Put the service(s) up"
	    @echo "\tdown - Pull the service(s) down"

.DEFAULT_GOAL := help

build:
	@docker-compose -f build/docker-compose.yml build

deploy:
	@docker-compose -f deploy/docker-compose.yml up -d

down:
	@docker-compose -f deploy/docker-compose.yml down
