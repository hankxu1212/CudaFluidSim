#pragma once

#include <string>
#include <cassert>
#include <functional>
#include <mutex>
#include <map>

#include "utils/Singleton.hpp"
#include "events/ApplicationEvent.hpp"
#include "layers/LayerStack.hpp"
#include "resources/Module.hpp"

int main(int argc, char** argv);

struct ApplicationCommandLineArgs
{
	int Count = 0;
	char** Args = nullptr;

	const char* operator[](int index) const
	{
		assert(index < Count);
		return Args[index];
	}
};

struct ApplicationSpecification
{
	std::string Name = "Cuda Fluid Simulator";
	uint32_t width = 1920;
	uint32_t height = 1080;
	ApplicationCommandLineArgs CommandLineArgs;
	
	enum class Culling { None, Frustum } Culling = Culling::None;
	std::optional<std::string> PhysicalDeviceName = std::nullopt;
	std::optional<std::string> CameraName = std::nullopt;
	std::optional<std::string> InitialScene = std::nullopt;

	bool alternativeApplication = false;
};

class Application : Singleton
{
public:
	static Application& Get() { return *s_Instance; }
	static Application* s_Instance;

public:
	Application(const ApplicationSpecification& specification);
	~Application();

	void SubmitToMainThread(const std::function<void()>& function);

	void OnEvent(Event& e);
	
	void PushLayer(Layer* layer);
	void PushOverlay(Layer* layer);

	void Close();

	static const char* GetArgs() { return s_Instance->GetSpecification().CommandLineArgs.Args[0]; }
	static const ApplicationSpecification& GetSpecification() { return s_Instance->m_Specification; }
	static const uint16_t GetFPS() { return s_Instance->m_FPS; }
	static const LayerStack& GetLayerStack() { return s_Instance->m_LayerStack; }
	static bool IsMinimized() { return s_Instance->m_Minimized; }


	inline static float ApplicationUpdateTime, ApplicationRenderTime;
	inline static bool StatsDirty = true;

private:
	void Run();

	void RunUpdate();
	void RunRender();

	void ExecuteMainThreadQueue();

	bool OnWindowClose(WindowCloseEvent& e);
	bool OnWindowIconfy(WindowIconfyEvent& e);
	bool OnWindowResize(WindowResizeEvent& e);
	void HandleWindowResizeComplete();
	bool isResizing = false;;

	void CreateModule(Module::RegistryMap::const_iterator it);
	void DestroyModule(TypeId id, Module::DestroyStage stage);
	void UpdateStage(Module::UpdateStage stage);
	void DestroyStage(Module::DestroyStage stage);

private:
	ApplicationSpecification			m_Specification;
	bool								m_Running = true;
	bool								m_Minimized = false;
	float								m_LastFrameTime = 0.0f;
	uint16_t							m_FPS = 0;
	uint16_t							m_FPS_Accumulator = 0;

	LayerStack												m_LayerStack;

	std::map<TypeId, std::unique_ptr<Module>>				m_Modules;
	std::map<Module::UpdateStage, std::vector<TypeId>>		m_ModuleStages;
	std::map<Module::DestroyStage, std::vector<TypeId>>		m_ModuleDestroyStages;

	std::vector<std::function<void()>>	m_MainThreadQueue;
	std::mutex							m_MainThreadQueueMutex;


private:
	friend int ::main(int argc, char** argv);
};