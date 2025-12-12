from .ppta import make

def create_openshape(config):
    if config.model.name == "PointBERT":
        model = make(config)
    else:
        raise NotImplementedError("Model %s not supported." % config.model.name)
    return model
