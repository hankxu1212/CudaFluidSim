#pragma once

#include "Renderer.hpp"

class UIRenderer : public Module::Registrar<UIRenderer>
{
	inline static const bool Registered = Register(
		UpdateStage::Pre,
		DestroyStage::Normal,
		Requires<Renderer>()
	);

public:
	UIRenderer();

	~UIRenderer();

	void Update();

	void RenderUIBegin();

	void RenderUIFinalize();

private:
	void SetTheme();
};

