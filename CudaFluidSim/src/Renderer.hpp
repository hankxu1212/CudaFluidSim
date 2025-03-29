#pragma once

#include "window/Window.hpp"

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
};

