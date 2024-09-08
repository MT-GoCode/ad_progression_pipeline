from prefect import context, flow

from ad_progression_pipeline.pipeline import end_to_end
from ad_progression_pipeline.pipeline.helpers import cli, context_handler

if __name__ == "__main__":
    args = cli.parse_args()
    context_handler.initialize_context(args.config)
    context.pipeline()
