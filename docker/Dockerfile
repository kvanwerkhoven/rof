FROM python:3.8-slim

RUN apt-get update \
    && apt install cron -y

# run on the first minute of every hour
RUN echo '1 * * * * python /src/ROF/rof_config.py >> /var/log/cron.log 2>&1' > /etc/cron.d/rof-cron && \
    chmod 0644 /etc/cron.d/rof-cron && \
    crontab /etc/cron.d/rof-cron && \
    touch /var/log/cron.log

# Create source and output directory
RUN mkdir -p /src /data /rof_maps

# copy source files and deps list
COPY requirements.txt /src
# install deps
RUN python -m pip install -r /src/requirements.txt
COPY ./ROF /src/ROF/

WORKDIR /src/ROF

# start cron and log to docker logs
ENTRYPOINT ["cron && tail -f /var/log/cron.log"]
