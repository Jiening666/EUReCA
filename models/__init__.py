from models.base_model import BaseModel


def create_recon_model(opt):
    """Instantiate the 2D→3D reconstruction model.

    Default (no flag): the original twoD2threeDModel.
    With --use_binned_aggr: the angular-binned two-stage aggregation variant
    (twoD2threeDBinningModel, K = --aggr_bins), which scales to hundreds of
    projections.  Imports are local so the default path never touches the
    binning module.
    """
    if getattr(opt, 'use_binned_aggr', False):
        from models.twod_to_threed_model_binning import twoD2threeDBinningModel
        return twoD2threeDBinningModel(opt)
    from models.twod_to_threed_model import twoD2threeDModel
    return twoD2threeDModel(opt)


def find_model_using_name(model_name):
    """Import the module "models/[model_name]_model.py".

    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    """Create a model given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance
