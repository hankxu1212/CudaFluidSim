#include "UIRenderer.h"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "implot.h"

#include "Solver.hpp"
#include "core/Application.hpp"
#include "Renderer.hpp"
#include "resources/Files.hpp"
#include "utils/Time.hpp"

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
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;         // Enable Docking
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;       // Enable Multi-Viewport / Platform Windows
    io.ConfigViewportsNoAutoMerge = true;
    io.ConfigViewportsNoTaskBarIcon = true;

    ImPlot::CreateContext();

    // Setup Dear ImGui style
    SetTheme();
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.5f, 0.9f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.6f, 1.0f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.1f, 0.4f, 0.8f, 1.0f));

    // When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
    ImGuiStyle& style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(Window::Get()->nativeWindow, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
}

UIRenderer::~UIRenderer()
{
    ImPlot::DestroyContext();

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
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

// Number of samples (averaged over 0.1 s windows) to hold for the graph.
constexpr int SOLVER_FRAME_HISTORY_COUNT = 120;

// Circular buffer that stores the averaged frame update times (in milliseconds).
static float g_SolverFrameTimes[SOLVER_FRAME_HISTORY_COUNT] = { 0.0f };
// Index used to insert the next averaged value in the circular buffer.
static int g_SolverFrameTimeIndex = 0;

// Accumulators for the 0.1-second averaging window.
static float g_AccumulatedFrameTime = 0.0f;
static float g_ElapsedTime = 0.0f;
static int   g_FrameCount = 0;

// Call this function each frame and pass the frame delta time (in seconds).
void UpdateSolverFrameTimeGraph(float deltaTime)
{
    // Retrieve the Solver's last update time (assumed to be in milliseconds).
    float currentFrameTime = Solver::Get()->LastFrameUpdateTime;

    // Accumulate the values.
    g_AccumulatedFrameTime += currentFrameTime;
    g_ElapsedTime += deltaTime;
    g_FrameCount++;

    // Check if we have reached (or exceeded) 0.1 seconds.
    if (g_ElapsedTime >= 0.1f)
    {
        // Compute the average frame time over this window.
        float averageFrameTime = g_AccumulatedFrameTime / static_cast<float>(g_FrameCount);

        // Store the average into the circular buffer.
        g_SolverFrameTimes[g_SolverFrameTimeIndex % SOLVER_FRAME_HISTORY_COUNT] = averageFrameTime;
        g_SolverFrameTimeIndex++;

        // Reset the accumulators for the next window.
        g_AccumulatedFrameTime = 0.0f;
        g_ElapsedTime = 0.0f;
        g_FrameCount = 0;
    }
}

// Resets the performance graph data.
void ResetSolverFrameTimeGraph()
{
    for (int i = 0; i < SOLVER_FRAME_HISTORY_COUNT; i++)
        g_SolverFrameTimes[i] = 0.0f;
    g_SolverFrameTimeIndex = 0;
    g_AccumulatedFrameTime = 0.0f;
    g_ElapsedTime = 0.0f;
    g_FrameCount = 0;
}

// Renders the performance graph window using ImPlot.
void RenderSolverPerformanceGraph()
{
    // Provide a button to reset the graph data.
    if (ImGui::Button("Reset Graph"))
    {
        ResetSolverFrameTimeGraph();
    }

    // Set up a plotting area with ImPlot.
    // The plot will display our averaged frame times with labeled axes.
    if (ImPlot::BeginPlot("Solver Performance", ImVec2(-1, 300)))
    {
        // Setup axes: X-axis is Time (s), Y-axis is Frame Update Time (ms).
        // Each sample represents a 0.1 second interval.  
        ImPlot::SetupAxes("Time (s)", "Frame Update Time (ms)", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);

        // Create an X-axis array based on the interval (0.1 seconds per sample).
        static float xs[SOLVER_FRAME_HISTORY_COUNT];
        for (int i = 0; i < SOLVER_FRAME_HISTORY_COUNT; i++)
        {
            xs[i] = i * 0.1;  // For example, sample 0 corresponds to time 0.0s, sample 1 to 0.1s, etc.
        }

        // Plot the averaged frame times.
        // Note: Even though our data is stored in a circular buffer, we pass the full buffer.
        // You might adapt this if you require a continuous plot.
        ImPlot::PlotLine("Avg Frame Time (ms)", xs, g_SolverFrameTimes, SOLVER_FRAME_HISTORY_COUNT);

        ImPlot::EndPlot();
    }

    // Optionally, display the most recent average value.
    int currentIndex = (g_SolverFrameTimeIndex - 1) % SOLVER_FRAME_HISTORY_COUNT;
    ImGui::Text("Last Avg Frame Time: %.2f ms", g_SolverFrameTimes[currentIndex]);
}

void UIRenderer::RenderUIFinalize()
{
    ImGui::Begin("Controls");
    {
        ImGui::SeparatorText("Application");
        Application::Get().OnImGuiRender();
        ImGui::Separator();
        
        ImGui::SeparatorText("Simulation");
        Solver::Get()->OnImGuiRender();
        if (Solver::Get()->Restart)
            ResetSolverFrameTimeGraph();
        ImGui::Separator();

        ImGui::SeparatorText("Rendering");
        Renderer::Get()->OnImGuiRender();
        ImGui::Separator();

        ImGui::SeparatorText("Profiling");
        UpdateSolverFrameTimeGraph(Time::DeltaTime);
        RenderSolverPerformanceGraph();
        ImGui::Separator();
    }
    ImGui::End();

    // Rendering
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(Window::Get()->nativeWindow, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    ImGuiIO& io = ImGui::GetIO(); (void)io;
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        GLFWwindow* backup_current_context = glfwGetCurrentContext();
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
        glfwMakeContextCurrent(backup_current_context);
    }
}

void UIRenderer::SetTheme()
{
    ImVec4* colors = ImGui::GetStyle().Colors;
    colors[ImGuiCol_Text] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
    colors[ImGuiCol_WindowBg] = ImVec4(0.05f, 0.05f, 0.05f, 0.80f);
    colors[ImGuiCol_ChildBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PopupBg] = ImVec4(0.19f, 0.19f, 0.19f, 0.92f);
    colors[ImGuiCol_Border] = ImVec4(0.19f, 0.19f, 0.19f, 0.29f);
    colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.24f);
    colors[ImGuiCol_FrameBg] = ImVec4(0.05f, 0.05f, 0.05f, 0.54f);
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.19f, 0.19f, 0.19f, 0.54f);
    colors[ImGuiCol_FrameBgActive] = ImVec4(0.20f, 0.22f, 0.23f, 1.00f);
    colors[ImGuiCol_TitleBg] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.06f, 0.06f, 0.06f, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_MenuBarBg] = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
    colors[ImGuiCol_ScrollbarBg] = ImVec4(0.05f, 0.05f, 0.05f, 0.54f);
    colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.34f, 0.34f, 0.34f, 0.54f);
    colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.40f, 0.40f, 0.40f, 0.54f);
    colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.56f, 0.56f, 0.56f, 0.54f);
    colors[ImGuiCol_CheckMark] = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
    colors[ImGuiCol_SliderGrab] = ImVec4(0.34f, 0.34f, 0.34f, 0.54f);
    colors[ImGuiCol_SliderGrabActive] = ImVec4(0.56f, 0.56f, 0.56f, 0.54f);
    colors[ImGuiCol_Button] = ImVec4(0.05f, 0.05f, 0.05f, 0.54f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.19f, 0.19f, 0.19f, 0.54f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.20f, 0.22f, 0.23f, 1.00f);
    colors[ImGuiCol_Header] = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.00f, 0.00f, 0.00f, 0.36f);
    colors[ImGuiCol_HeaderActive] = ImVec4(0.20f, 0.22f, 0.23f, 0.33f);
    colors[ImGuiCol_Separator] = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
    colors[ImGuiCol_SeparatorHovered] = ImVec4(0.44f, 0.44f, 0.44f, 0.29f);
    colors[ImGuiCol_SeparatorActive] = ImVec4(0.40f, 0.44f, 0.47f, 1.00f);
    colors[ImGuiCol_ResizeGrip] = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
    colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.44f, 0.44f, 0.44f, 0.29f);
    colors[ImGuiCol_ResizeGripActive] = ImVec4(0.40f, 0.44f, 0.47f, 1.00f);
    colors[ImGuiCol_Tab] = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
    colors[ImGuiCol_TabHovered] = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
    colors[ImGuiCol_TabActive] = ImVec4(0.20f, 0.20f, 0.20f, 0.36f);
    colors[ImGuiCol_TabUnfocused] = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
    colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
    colors[ImGuiCol_DockingPreview] = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
    colors[ImGuiCol_DockingEmptyBg] = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_PlotLines] = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_PlotHistogram] = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_TableHeaderBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
    colors[ImGuiCol_TableBorderStrong] = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
    colors[ImGuiCol_TableBorderLight] = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
    colors[ImGuiCol_TableRowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_TableRowBgAlt] = ImVec4(1.00f, 1.00f, 1.00f, 0.06f);
    colors[ImGuiCol_TextSelectedBg] = ImVec4(0.20f, 0.22f, 0.23f, 1.00f);
    colors[ImGuiCol_DragDropTarget] = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
    colors[ImGuiCol_NavHighlight] = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 0.00f, 0.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg] = ImVec4(1.00f, 0.00f, 0.00f, 0.20f);
    colors[ImGuiCol_ModalWindowDimBg] = ImVec4(1.00f, 0.00f, 0.00f, 0.35f);

    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowPadding = ImVec2(8.00f, 8.00f);
    style.FramePadding = ImVec2(5.00f, 2.00f);
    style.CellPadding = ImVec2(6.00f, 6.00f);
    style.ItemSpacing = ImVec2(6.00f, 6.00f);
    style.ItemInnerSpacing = ImVec2(6.00f, 6.00f);
    style.TouchExtraPadding = ImVec2(0.00f, 0.00f);
    style.IndentSpacing = 25;
    style.ScrollbarSize = 15;
    style.GrabMinSize = 10;
    style.WindowBorderSize = 1;
    style.ChildBorderSize = 1;
    style.PopupBorderSize = 1;
    style.FrameBorderSize = 1;
    style.TabBorderSize = 1;
    style.WindowRounding = 7;
    style.ChildRounding = 4;
    style.FrameRounding = 3;
    style.PopupRounding = 4;
    style.ScrollbarRounding = 9;
    style.GrabRounding = 3;
    style.LogSliderDeadzone = 4;
    style.TabRounding = 4;

    // Load Fonts
    const std::string pathRelativeToExecutable = "../../CudaFluidSim/src/";
    std::string path = Files::Path(pathRelativeToExecutable + "fonts/SourceSans3-Regular.ttf");
    ImFont* font = ImGui::GetIO().Fonts->AddFontFromFileTTF(path.c_str(), 24.0);
    IM_ASSERT(font != nullptr);
}

