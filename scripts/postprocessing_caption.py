from components import scripts_postprocessing, deepbooru, shared
from ui import ui_components
import gradio as gr


class ScriptPostprocessingCeption(scripts_postprocessing.ScriptPostprocessing):
    name = "Caption"
    order = 4000

    def ui(self):
        with ui_components.InputAccordion(False, label="Caption") as enable:
            option = gr.CheckboxGroup(value=["Deepbooru"], choices=["Deepbooru", "BLIP"], show_label=False)

        return {
            "enable": enable,
            "option": option,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, enable, option):
        if not enable:
            return

        captions = [pp.caption]

        if "Deepbooru" in option:
            captions.append(deepbooru.model.tag(pp.image))

        if "BLIP" in option:
            captions.append(shared.interrogator.generate_caption(pp.image))

        pp.caption = ", ".join([x for x in captions if x])
