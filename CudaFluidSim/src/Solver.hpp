#pragma once

#include "window/Window.hpp"

class Solver : public Module::Registrar<Solver>
{
	inline static const bool Registered = Register(
		UpdateStage::Pre, 
		DestroyStage::Normal,
		Requires<Window>()
	);

public:
	Solver();

	virtual ~Solver();

	void Update();
};

