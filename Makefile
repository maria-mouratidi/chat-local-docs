ingest:
	uv run src/main.py ingest $(DIR)

query:
	uv run src/main.py query "$(Q)"
