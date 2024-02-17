import sys

from components.shared_cmd_options import cmd_opts


def realesrgan_models_names():
    import component.realesrgan_model
    return [x.name for x in component.realesrgan_model.get_realesrgan_models(None)]


def postprocessing_scripts():
    import component.scripts

    return component.scripts.scripts_postproc.scripts


def sd_vae_items():
    import component.sd_vae

    return ["Automatic", "None"] + list(component.sd_vae.vae_dict)


def refresh_vae_list():
    import component.sd_vae

    component.sd_vae.refresh_vae_list()


def cross_attention_optimizations():
    import component.sd_hijack

    return ["Automatic"] + [x.title() for x in component.sd_hijack.optimizers] + ["None"]


def sd_unet_items():
    import component.sd_unet

    return ["Automatic"] + [x.label for x in component.sd_unet.unet_options] + ["None"]


def refresh_unet_list():
    import component.sd_unet

    component.sd_unet.list_unets()


def list_checkpoint_tiles(use_short=False):
    import component.sd_models
    return component.sd_models.checkpoint_tiles(use_short)


def refresh_checkpoints():
    import component.sd_models
    return component.sd_models.list_models()


def list_samplers():
    import component.sd_samplers
    return component.sd_samplers.all_samplers


def reload_hypernetworks():
    from components.hypernetworks import hypernetwork
    from components import shared

    shared.hypernetworks = hypernetwork.list_hypernetworks(cmd_opts.hypernetwork_dir)


def get_infotext_names():
    from components import generation_parameters_copypaste, shared
    res = {}

    for info in shared.opts.data_labels.values():
        if info.infotext:
            res[info.infotext] = 1

    for tab_data in generation_parameters_copypaste.paste_fields.values():
        for _, name in tab_data.get("fields") or []:
            if isinstance(name, str):
                res[name] = 1

    return list(res)


ui_reorder_categories_builtin_items = [
    "prompt",
    "image",
    "inpaint",
    "sampler",
    "accordions",
    "checkboxes",
    "dimensions",
    "cfg",
    "denoising",
    "seed",
    "batch",
    "override_settings",
]


def ui_reorder_categories():
    from components import scripts

    yield from ui_reorder_categories_builtin_items

    sections = {}
    for script in scripts.scripts_txt2img.scripts + scripts.scripts_img2img.scripts:
        if isinstance(script.section, str) and script.section not in ui_reorder_categories_builtin_items:
            sections[script.section] = 1

    yield from sections

    yield "scripts"


class Shared(sys.module[__name__].__class__):
    """
    this class is here to provide sd_model field as a property, so that it can be created and loaded on demand rather than
    at program startup.
    """

    sd_model_val = None

    @property
    def sd_model(self):
        import component.sd_models

        return component.sd_models.model_data.get_sd_model()

    @sd_model.setter
    def sd_model(self, value):
        import component.sd_models

        component.sd_models.model_data.set_sd_model(value)


sys.module['modules.shared'].__class__ = Shared
