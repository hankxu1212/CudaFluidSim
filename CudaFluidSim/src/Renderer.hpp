#pragma once

#include "window/Window.hpp"
#include "resources/ResourceManager.h"
#include "renderer/MetaballRenderer.hpp"

class Renderer : public Module::Registrar<Renderer>
{
	inline static const bool Registered = Register(
		UpdateStage::Render,
		DestroyStage::Post,
		Requires<Window>()
	);

public:
	Renderer();

	virtual ~Renderer();

	void Update();

	void OnImGuiRender();

	bool Disabled = false;
	bool UseBlur = false;
	bool UseMetaballRendering = false;

private:
	Shader shader_splat;
	unsigned int VAO, VBO;

	std::unique_ptr<MetaballRenderer> metaballRenderer;
};




