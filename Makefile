deploy:
	uv run modal deploy modal_app.py

ingest:
	uv run src/main.py ingest $(DIR)

query:
	uv run src/main.py query "$(Q)"
