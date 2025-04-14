#pragma once

#include "window/Window.hpp"
#include "resources/ResourceManager.h"

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

private:
	Shader shader;
	Shader shader_splat;
	unsigned int VAO, VBO;
};




