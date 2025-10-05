import { useState } from "react";

function ChatPage() {
  const [messages, setMessages] = useState([
    { sender: "bot", text: "Hi, how can I help you today?" },
    { sender: "user", text: "I want to know about React." },
  ]);
  const [input, setInput] = useState("");

  const handleSend = () => {
    if (!input.trim()) return;
    setMessages([...messages, { sender: "user", text: input }]);
    setInput("");
    setTimeout(() => {
      setMessages((prev) => [...prev, { sender: "bot", text: "Got it! Let me explain React for you." }]);
    }, 800);
  };

  return (
    <div className="flex flex-col h-[calc(100vh-64px)]">
      <header className="bg-[#F5F4E9] p-4 text-center font-semibold">
        Chat with EduLoop
      </header>

      <main className="flex-1 overflow-y-auto p-4 space-y-3 px-30">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex ${msg.sender === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className="p-3 rounded-2xl max-w-xs text-sm md:text-base"
            >
              {msg.text}
            </div>
          </div>
        ))}
      </main>

      <footer className="p-3 bg-white flex items-center space-x-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder="Type a message..."
          className="flex-1 border rounded-full px-5 ml-30 py-2 focus:outline-none mb-10"
        />
        <button
          onClick={handleSend}
          className="bg-[#F5F4E9] px-4 py-2 rounded-full mr-25 mb-10 border border-[#ccc] hover:bg-[#eee]"
        >
          Send
        </button>
      </footer>
    </div>
  );
}

export default ChatPage