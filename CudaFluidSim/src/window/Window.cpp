#include <format>

#include "Window.hpp"

#include "events/KeyEvent.hpp"
#include "events/MouseEvent.hpp"
#include "GLFW/glfw3native.h"

static uint8_t s_GLFWWindowCount = 0;

static void glfwSetWindowCenter(GLFWwindow* window) {
	// Get window m_Position and size
	int window_x, window_y;
	glfwGetWindowPos(window, &window_x, &window_y);

	int window_width, window_height;
	glfwGetWindowSize(window, &window_width, &window_height);

	// Halve the window size and use it to adjust the window m_Position to the center of the window
	window_width /= 2;
	window_height /= 2;

	window_x += window_width;
	window_y += window_height;

	// Get the list of monitors
	int monitors_length;
	GLFWmonitor** monitors = glfwGetMonitors(&monitors_length);

	if (monitors == NULL) {
		// Got no monitors back
		return;
	}

	// Figure out which monitor the window is in
	GLFWmonitor* owner = NULL;
	int owner_x = 0, owner_y = 0, owner_width = 0, owner_height = 0;

	for (int i = 0; i < monitors_length; i++) {
		// Get the monitor m_Position
		int monitor_x, monitor_y;
		glfwGetMonitorPos(monitors[i], &monitor_x, &monitor_y);

		// Get the monitor size from its video mode
		int monitor_width, monitor_height;
		GLFWvidmode* monitor_vidmode = (GLFWvidmode*)glfwGetVideoMode(monitors[i]);

		if (monitor_vidmode == NULL) {
			// Video mode is required for width and height, so skip this monitor
			continue;

		}
		else {
			monitor_width = monitor_vidmode->width;
			monitor_height = monitor_vidmode->height;
		}

		// Set the owner to this monitor if the center of the window is within its bounding box
		if ((window_x > monitor_x && window_x < (monitor_x + monitor_width)) && (window_y > monitor_y && window_y < (monitor_y + monitor_height))) {
			owner = monitors[i];

			owner_x = monitor_x;
			owner_y = monitor_y;

			owner_width = monitor_width;
			owner_height = monitor_height;
		}
	}

	if (owner != NULL) {
		// Set the window m_Position to the center of the owner monitor
		glfwSetWindowPos(window, owner_x + (owner_width >> 1) - window_width, owner_y + (owner_height >> 1) - window_height);
	}
}

static void GLFWErrorCallback(int error, const char* description)
{
	std::cerr << "GLFW Error: " << error << ": " << description << std::endl;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	glViewport(0, 0, width, height);
}

Window::Window()
{
	const ApplicationSpecification& props = Application::GetSpecification();

	m_Data.Title = props.Name;
	m_Data.Width = props.width;
	m_Data.Height = props.height;

	if (s_GLFWWindowCount == 0)
	{
		assert(glfwInit() && "[glfw]: Could not initialize GLFW!");
		glfwSetErrorCallback(GLFWErrorCallback);
		s_GLFWWindowCount++;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	nativeWindow = glfwCreateWindow((int)props.width, (int)props.height, m_Data.Title.c_str(), nullptr, nullptr);

	glfwSetWindowUserPointer(nativeWindow, &m_Data);
	m_Data.VSync = true;

	// event handlings, set GLFW callbacks
	{
		glfwSetWindowSizeCallback(nativeWindow, [](GLFWwindow* window, int width, int height)
			{
				if (width <= 0 || height <= 0) return;
				WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
				data.Width = width;
				data.Height = height;

				WindowResizeEvent event(width, height);
				data.EventCallback(event);
			});

		glfwSetWindowIconifyCallback(nativeWindow, [](GLFWwindow* window, int iconfied)
			{
				WindowIconfyEvent event(iconfied);
				WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
				data.EventCallback(event);
			});

		glfwSetWindowCloseCallback(nativeWindow, [](GLFWwindow* window)
			{
				WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
				WindowCloseEvent event;
				data.EventCallback(event);
			});

		glfwSetKeyCallback(nativeWindow, [](GLFWwindow* window, int key, int scancode, int action, int mods)
			{
				WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

				switch (action)
				{
				case GLFW_PRESS:
				{
					KeyPressedEvent event((KeyCode)key, false);
					data.EventCallback(event);
					break;
				}
				case GLFW_RELEASE:
				{
					KeyReleasedEvent event((KeyCode)key);
					data.EventCallback(event);
					break;
				}
				case GLFW_REPEAT:
				{
					KeyPressedEvent event((KeyCode)key, true);
					data.EventCallback(event);
					break;
				}
				}
			});

		glfwSetCharCallback(nativeWindow, [](GLFWwindow* window, unsigned int keycode)
			{
				WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

				KeyTypedEvent event((KeyCode)keycode);
				data.EventCallback(event);
			});

		glfwSetMouseButtonCallback(nativeWindow, [](GLFWwindow* window, int button, int action, int mods)
			{
				WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

				switch (action)
				{
				case GLFW_PRESS:
				{
					MouseButtonPressedEvent event((MouseCode)button);
					data.EventCallback(event);
					break;
				}
				case GLFW_RELEASE:
				{
					MouseButtonReleasedEvent event((MouseCode)button);
					data.EventCallback(event);
					break;
				}
				}
			});

		glfwSetScrollCallback(nativeWindow, [](GLFWwindow* window, double xOffset, double yOffset)
			{
				WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

				MouseScrolledEvent event((float)xOffset, (float)yOffset);
				data.EventCallback(event);
			});

		glfwSetCursorPosCallback(nativeWindow, [](GLFWwindow* window, double xPos, double yPos)
			{
				WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

				MouseMovedEvent event((float)xPos, (float)yPos);
				data.EventCallback(event);
			});

	}

	glfwSetWindowCenter(nativeWindow);

	glfwMakeContextCurrent(nativeWindow);

	glfwSetFramebufferSizeCallback(nativeWindow, framebuffer_size_callback);
}

Window::~Window()
{
	std::cout << "Destroyed window module\n" << std::flush;

	glfwDestroyWindow(nativeWindow);
	--s_GLFWWindowCount;

	if (s_GLFWWindowCount == 0)
	{
		glfwTerminate();
	}
}

void Window::Update()
{
	// poll events on pre stage
	glfwPollEvents();
}
