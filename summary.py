def summary():

  from model import build_model
  from args import get_args

  args = get_args()

  model = build_model(args.model_name,(378,504,3))

  model.summary()

summary()