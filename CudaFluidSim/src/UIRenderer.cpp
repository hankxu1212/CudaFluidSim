#include "UIRenderer.h"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

UIRenderer::UIRenderer()
{
    // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
    // GL ES 2.0 + GLSL 100 (WebGL 1.0)
    const char* glsl_version = "#version 100";
#elif defined(IMGUI_IMPL_OPENGL_ES3)
    // GL ES 3.0 + GLSL 300 es (WebGL 2.0)
    const char* glsl_version = "#version 300 es";
#elif defined(__APPLE__)
    // GL 3.2 + GLSL 150
    const char* glsl_version = "#version 150";
#else
    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
#endif

    // initialize imgui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(Window::Get()->nativeWindow, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
}

UIRenderer::~UIRenderer()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void UIRenderer::Update()
{
    RenderUIBegin();
}

void UIRenderer::RenderUIBegin()
{
    //ImGui_ImplOpenGL3_NewFrame();
    //ImGui_ImplGlfw_NewFrame();
    //ImGui::NewFrame();
}

void UIRenderer::RenderUIFinalize()
{
    //bool show_demo_window = true;
    //if (show_demo_window)
    //    ImGui::ShowDemoWindow(&show_demo_window);

    //ImGui::Begin("Another Window", &show_demo_window);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
    //ImGui::Text("Hello from another window!");
    //if (ImGui::Button("Close Me"))
    //    show_demo_window = false;
    //ImGui::End();

    // Rendering
    //ImGui::Render();
    //int display_w, display_h;
    //glfwGetFramebufferSize(Window::Get()->nativeWindow, &display_w, &display_h);
    //glViewport(0, 0, display_w, display_h);
    //ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
