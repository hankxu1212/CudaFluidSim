#include "Renderer.hpp"
#include "math/Math.hpp"
#include "resources/Files.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "window/Window.hpp"

#include "SpatialHashTable.hpp"
#include "Solver.hpp"

#include "UIRenderer.h"
#include "imgui.h"

Renderer::Renderer()
{
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
    }

	const std::string pathRelativeToExecutable = "../../CudaFluidSim/src/";

	// load shaders
	std::string vertPath = Files::Path(pathRelativeToExecutable + "shaders/particle.vert");
	std::string fragPath = Files::Path(pathRelativeToExecutable + "shaders/particle.frag");
    shader = ResourceManager::LoadShader(vertPath.c_str(), fragPath.c_str(), nullptr, "particle");

	// configure projection
	glm::mat4 projection = glm::ortho(0.0f, 
		static_cast<float>(Window::Get()->m_Data.Width), 
		static_cast<float>(Window::Get()->m_Data.Height)
		, 0.0f, -1.0f, 1.0f);

    shader.Use().SetMatrix4("projection", projection);

    // set up mesh and attribute properties
    unsigned int VBO;
    float particle_quad[] = {
        0.0f, 1.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,

        0.0f, 1.0f, 0.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 0.0f, 1.0f, 0.0f
    };
    glGenVertexArrays(1, &this->VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(this->VAO);

    // fill mesh buffer
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(particle_quad), particle_quad, GL_STATIC_DRAW);
    
    // set mesh attributes
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glBindVertexArray(0);
}

Renderer::~Renderer()
{
    glDeleteVertexArrays(1, &VAO);
    glDeleteProgram(shader.ID);
}

// update is called in Render stage
void Renderer::Update()
{
    glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    if (Disabled) 
    {
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        UIRenderer::Get()->RenderUIFinalize();
        glfwSwapBuffers(Window::Get()->nativeWindow);
        return;
    }

    if (Application::GetSpecification().headless)
        return;

    // render particles
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    this->shader.Use();
    const auto& particles = Solver::Get()->m_Particles;
    for (const auto& particle : particles)
    {
        this->shader.SetFloat("particleRGB", 0);
        this->shader.SetVector2f("center", particle.position);
        glBindVertexArray(this->VAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);
    }

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    UIRenderer::Get()->RenderUIFinalize();

    glfwSwapBuffers(Window::Get()->nativeWindow);
}

void Renderer::OnImGuiRender()
{
    if (ImGui::Button(Disabled ? "Enable Renderer" : "Disable Renderer"))
        Disabled = !Disabled;

    if (Disabled)
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Renderer: Disabled");
    else
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Renderer: Enabled");
}
