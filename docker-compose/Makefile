########################################
## Redpanda management
########################################

start-redpanda: 
	docker compose -f redpanda.yml up -d

stop-redpanda:
	docker compose -f redpanda.yml down

add-one-partition-to-trades-topic:
	docker compose -f redpanda.yml exec redpanda rpk topic add-partitions trades --num 1


########################################
## Elasticsearch
########################################

start-elasticsearch:
	docker compose -f elasticsearch.yml up -d

stop-elasticsearch:
	docker compose -f elasticsearch.yml down