document.addEventListener('DOMContentLoaded', () => {
    const chatContainer = document.getElementById('chat-container');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    const historyList = document.getElementById('history-list');
    const newChatBtn = document.getElementById('new-chat-btn');

    // Mobile Menu Logic
    const mobileMenuBtn = document.getElementById('mobile-menu-btn');
    const sidebar = document.querySelector('.sidebar');

    const sunIcon = document.querySelector('.sun-icon');
    const moonIcon = document.querySelector('.moon-icon');

    // State
    let chats = JSON.parse(localStorage.getItem('fiber_gpt_chats')) || [];
    let currentChatIndex = null;
    let theme = localStorage.getItem('fiber_gpt_theme') || 'light';

    // Initialize
    applyTheme(theme);
    renderHistory();
    if (chats.length > 0) {
        loadChat(0); // Load first chat by default
    } else {
        startNewChat();
    }

    // Mobile Menu
    mobileMenuBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevent document click from closing it immediately
        console.log("Mobile menu clicked");
        sidebar.classList.toggle('open');
        document.body.classList.toggle('sidebar-open');
    });

    // Close sidebar when clicking outside on mobile
    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 768 &&
            sidebar.classList.contains('open') &&
            !sidebar.contains(e.target) &&
            !mobileMenuBtn.contains(e.target)) {
            sidebar.classList.remove('open');
            document.body.classList.remove('sidebar-open');
        }
    });

    // Theme Toggle
    themeToggleBtn.addEventListener('click', () => {
        theme = theme === 'light' ? 'dark' : 'light';
        localStorage.setItem('fiber_gpt_theme', theme);
        applyTheme(theme);
    });

    function applyTheme(t) {
        document.documentElement.setAttribute('data-theme', t);
        updateThemeIcon(t);
    }

    function updateThemeIcon(t) {
        // Logo Logic
        const logo = document.getElementById('app-logo');
        if (logo) {
            logo.src = t === 'dark' ? '/static/fiber-gpt-logo-white.png' : '/static/fiber-gpt-logo-black.png';
        }

        if (t === 'dark') {
            sunIcon.classList.remove('hidden');
            moonIcon.classList.add('hidden');
        } else {
            sunIcon.classList.add('hidden');
            moonIcon.classList.remove('hidden');
        }
    }

    // Input Logic
    function toggleSendButton() {
        if (chatInput.value.trim() !== "") {
            sendBtn.removeAttribute('disabled');
            sendBtn.style.opacity = "1";
            sendBtn.style.cursor = "pointer";
        } else {
            sendBtn.setAttribute('disabled', 'true');
            sendBtn.style.opacity = "0.2";
            sendBtn.style.cursor = "default";
        }
    }

    // Auto-resize input
    chatInput.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';

        toggleSendButton();
    });

    // Handle Enter key
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // Prevent new line
            if (chatInput.value.trim() !== '') {
                sendMessage();
                // Reset height
                chatInput.style.height = 'auto';
            }
        }
    });

    sendBtn.addEventListener('click', () => {
        if (chatInput.value.trim() !== '') {
            sendMessage();
            chatInput.style.height = 'auto';
        }
    });

    async function sendMessage() {
        const text = chatInput.value.trim();
        if (!text) return;

        // Clear welcome message if it exists
        const welcome = document.querySelector('.welcome-message');
        if (welcome) welcome.remove();

        // User Message
        addMessage(text, 'user');
        chatInput.value = '';
        toggleSendButton();

        // Construct Context
        let context = [];
        if (currentChatIndex !== null && chats[currentChatIndex]) {
            context = chats[currentChatIndex].messages.map(msg => ({
                role: msg.sender === 'user' ? 'user' : 'assistant',
                content: msg.text
            }));
        }
        context.push({ role: 'user', content: text });

        // Bot Response (Placeholder)
        const botMsgContent = addMessage('', 'bot', true); // Empty initially, kept loading
        let fullResponse = "";

        try {
            const response = await fetch('/chat/completions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    messages: context,
                    temperature: 0.7,
                    max_tokens: 512
                })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Server Error');
            }

            // Read the stream
            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const jsonStr = line.substring(6).trim();
                        if (jsonStr === "[DONE]" || (jsonStr.includes('"done": true'))) break;

                        try {
                            const data = JSON.parse(jsonStr);
                            if (data.token) {
                                fullResponse += data.token;
                                botMsgContent.textContent = fullResponse;
                                // Auto scroll
                                chatContainer.scrollTop = chatContainer.scrollHeight;
                            }
                        } catch (e) {
                            console.error("Error parsing SSE chunk", e);
                        }
                    }
                }
            }

            botMsgContent.parentElement.classList.remove('loading');

            // Save conversation
            saveChat({ text: text, sender: 'user' });
            saveChat({ text: fullResponse, sender: 'bot' });

        } catch (error) {
            botMsgContent.textContent = "Error: " + error.message;
            botMsgContent.style.color = 'red';
            botMsgContent.parentElement.classList.remove('loading');
        }
    }

    function addMessage(text, sender, isLoading = false) {
        const msgDiv = document.createElement('div');
        msgDiv.classList.add('message', sender);
        if (isLoading) msgDiv.classList.add('loading');

        const contentDiv = document.createElement('div');
        contentDiv.classList.add('message-content');
        contentDiv.textContent = text;

        msgDiv.appendChild(contentDiv);
        chatContainer.appendChild(msgDiv);

        // Auto scroll
        chatContainer.scrollTop = chatContainer.scrollHeight;

        return contentDiv;
    }

    // Chat History Logic
    function saveChat(msg) {
        if (currentChatIndex === null) {
            // New Chat
            const newChat = {
                title: msg.text.slice(0, 30) + '...',
                messages: []
            };
            chats.unshift(newChat); // Add to beginning
            currentChatIndex = 0;
        }

        // Add message to current chat
        chats[currentChatIndex].messages.push(msg);
        localStorage.setItem('fiber_gpt_chats', JSON.stringify(chats));
        renderHistory();
    }

    function renderHistory() {
        historyList.innerHTML = '';

        chats.forEach((chat, index) => {
            const item = document.createElement('div');
            item.classList.add('history-item');
            if (index === currentChatIndex) item.classList.add('active');
            item.textContent = chat.title || 'New Chat';
            item.addEventListener('click', () => loadChat(index));
            historyList.appendChild(item);
        });
    }

    function loadChat(index) {
        currentChatIndex = index;
        const chat = chats[index];

        // Clear UI
        chatContainer.innerHTML = '';

        // Render Messages
        chat.messages.forEach(msg => {
            addMessage(msg.text, msg.sender);
        });

        renderHistory();

        // Close sidebar on mobile
        if (window.innerWidth <= 768) {
            sidebar.classList.remove('open');
            document.body.classList.remove('sidebar-open');
        }

        // Focus input
        setTimeout(() => chatInput.focus(), 50);
    }

    // New Chat
    newChatBtn.addEventListener('click', startNewChat);

    function startNewChat() {
        currentChatIndex = null;
        chatContainer.innerHTML = `
            <div class="welcome-message">
                <h1>?</h1>
            </div>
        `;
        renderHistory();

        // Close sidebar on mobile
        if (window.innerWidth <= 768) {
            sidebar.classList.remove('open');
            document.body.classList.remove('sidebar-open');
        }

        setTimeout(() => chatInput.focus(), 50);
    }
});
