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

#include <omp.h>

//--------------------------------------------------------------
// Post-Processing: Full-Screen Quad with Blur
//--------------------------------------------------------------
class PostProcessor {
public:
    PostProcessor() : quadVAO(0), quadVBO(0) {
        Initialize();
    }
    ~PostProcessor() {
        Cleanup();
    }

    void Initialize() {

        const std::string pathRelativeToExecutable = "../../CudaFluidSim/src/";
        std::string vertPath = Files::Path(pathRelativeToExecutable + "shaders/quad.vert");
        std::string fragPath = Files::Path(pathRelativeToExecutable + "shaders/blur.frag");
        blurShader = ResourceManager::LoadShader(vertPath.c_str(), fragPath.c_str(), nullptr, "blur");

        // Set up a full-screen quad.
        float quadVertices[] = {
            // positions   // texCoords
            -1.0f,  1.0f,   0.0f, 1.0f,
            -1.0f, -1.0f,   0.0f, 0.0f,
             1.0f, -1.0f,   1.0f, 0.0f,

            -1.0f,  1.0f,   0.0f, 1.0f,
             1.0f, -1.0f,   1.0f, 0.0f,
             1.0f,  1.0f,   1.0f, 1.0f
        };

        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);
        glBindVertexArray(quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
        // positions
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        // texcoords
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    // Render the blurred scene from the given texture.
    void Render(unsigned int texture) {
        blurShader.Use();
        blurShader.SetInteger("scene", 0);
        blurShader.SetFloat("blurSize", 5.0f); // Adjust this value to control blur strength.
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);

        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);
    }

    void Cleanup() {
        if (quadVBO) { glDeleteBuffers(1, &quadVBO); quadVBO = 0; }
        if (quadVAO) { glDeleteVertexArrays(1, &quadVAO); quadVAO = 0; }
        glDeleteProgram(blurShader.ID);
    }

private:
    unsigned int quadVAO, quadVBO;
    Shader blurShader;
};

//--------------------------------------------------------------
// Framebuffer Object (FBO) Setup for Post-Processing
//--------------------------------------------------------------
class FBO {
public:
    FBO(int width, int height)
        : width(width), height(height), fbo(0), textureColorBuffer(0)
    {
        Initialize();
    }
    ~FBO() { Cleanup(); }

    void Bind() { glBindFramebuffer(GL_FRAMEBUFFER, fbo); }
    void Unbind() { glBindFramebuffer(GL_FRAMEBUFFER, 0); }
    unsigned int GetTexture() const { return textureColorBuffer; }

private:
    int width, height;
    unsigned int fbo, textureColorBuffer;

    void Initialize() {
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);

        // Create texture to hold the framebuffer color output.
        glGenTextures(1, &textureColorBuffer);
        glBindTexture(GL_TEXTURE_2D, textureColorBuffer);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorBuffer, 0);

        // Check if framebuffer is complete.
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void Cleanup() {
        if (textureColorBuffer) { glDeleteTextures(1, &textureColorBuffer); textureColorBuffer = 0; }
        if (fbo) { glDeleteFramebuffers(1, &fbo); fbo = 0; }
    }
};

Renderer::Renderer()
{
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
    }

	const std::string pathRelativeToExecutable = "../../CudaFluidSim/src/";

    std::string vertPath_splat = Files::Path(pathRelativeToExecutable + "shaders/particle_splat.vert");
    std::string fragPath_splat = Files::Path(pathRelativeToExecutable + "shaders/particle_splat.frag");
    shader_splat = ResourceManager::LoadShader(vertPath_splat.c_str(), fragPath_splat.c_str(), nullptr, "particle_splat");

	// configure projection
	glm::mat4 projection = glm::ortho(0.0f, 
		static_cast<float>(Window::Get()->m_Data.Width), 
		static_cast<float>(Window::Get()->m_Data.Height)
		, 0.0f, -1.0f, 1.0f);

    //shader.Use().SetMatrix4("projection", projection);
    shader_splat.Use().SetMatrix4("projection", projection);

    // Create and bind the VAO.
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    // Create the VBO to hold particle positions.
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, NUM_PARTICLES * sizeof(Particle), nullptr, GL_DYNAMIC_DRAW);

    // Setup the vertex attribute pointer for particle positions.
    // Assume location 0 will be used for the position.
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    metaballRenderer = std::make_unique<MetaballRenderer>(Window::Get()->m_Data.Width, Window::Get()->m_Data.Height);
}

Renderer::~Renderer()
{
    // Delete the VBO and VAO.
    if (VBO) {
        glDeleteBuffers(1, &VBO);
        VBO = 0;
    }
    if (VAO) {
        glDeleteVertexArrays(1, &VAO);
        VAO = 0;
    }

    //glDeleteProgram(shader.ID);
    glDeleteProgram(shader_splat.ID);
}

// update is called in Render stage
void Renderer::Update()
{
    glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    if (Disabled)
    {
        UIRenderer::Get()->RenderUIFinalize();
        glfwSwapBuffers(Window::Get()->nativeWindow);
        return;
    }
    else if (Application::GetSpecification().headless) 
    {
        UIRenderer::Get()->RenderUIFinalize();
        glfwSwapBuffers(Window::Get()->nativeWindow);
        return;
    }

    static FBO fbo(Window::Get()->m_Data.Width, Window::Get()->m_Data.Height);
    static PostProcessor postProcessor;

    if (UseBlur)
        fbo.Bind();

    // do particle rendering here
    {
        const auto& particles = Solver::Get()->m_Particles;

        if (UseMetaballRendering)
            metaballRenderer->Render(particles.data(), NUM_PARTICLES);
        else 
        {
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferSubData(GL_ARRAY_BUFFER, 0, NUM_PARTICLES * sizeof(Particle), particles.data());
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            shader_splat.Use();

            // Optionally, send any additional uniform such as splat size.
            // For example: shader.SetFloat("pointSize", someValue);

            // Bind the VAO containing the particle positions.
            glBindVertexArray(VAO);

            // Enable using programmable point sizes if needed.
            glEnable(GL_PROGRAM_POINT_SIZE);

            // Draw all particles as points.
            glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);

            glBindVertexArray(0);
        }
    }

    // Render the blurred texture to the screen.
    if (UseBlur) 
    {
        fbo.Unbind();
        
        glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        postProcessor.Render(fbo.GetTexture());
    }

    // UI rendering and buffer swapping.
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

    if (ImGui::Button(UseBlur ? "Disable Blur" : "Use Blur"))
        UseBlur = !UseBlur;

    if (ImGui::Button(UseMetaballRendering ? "Disable Metaball Rendering" : "Use Metaball Rendering"))
        UseMetaballRendering = !UseMetaballRendering;

    if (UseMetaballRendering)
    {
        ImGui::DragFloat("Metaball Threshold", &metaballRenderer->MetaballThreshold, 0.01f, 0.0f, 10.0f);
    }
}
