import { useState } from "react";
import { Send, Mic } from "lucide-react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { cn } from "@/lib/utils";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
}

const ChatInterface = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isListening, setIsListening] = useState(false);

  const handleSend = () => {
    if (!input.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");

    // Simulate assistant response
    setTimeout(() => {
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "I'm ZENITH, your AI assistant. I'm currently in development mode. Full AI integration coming soon!",
      };
      setMessages((prev) => [...prev, assistantMessage]);
    }, 1000);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSend();
    }
  };

  const toggleVoiceInput = () => {
    setIsListening(!isListening);
    // Voice input functionality to be implemented
  };

  return (
    <div className="w-full max-w-4xl mx-auto px-4">
      {/* Messages Container */}
      <div className="mb-6 space-y-4 max-h-[400px] overflow-y-auto">
        {messages.length === 0 ? (
          <div className="text-center text-muted-foreground py-8">
            <p className="text-lg">Start a conversation with ZENITH</p>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={cn(
                "flex animate-fade-in",
                message.role === "user" ? "justify-end" : "justify-start"
              )}
            >
              <div
                className={cn(
                  "max-w-[70%] px-6 py-3 rounded-3xl glass-effect",
                  message.role === "user"
                    ? "bg-primary text-primary-foreground"
                    : "bg-card text-card-foreground"
                )}
              >
                <p className="text-sm leading-relaxed">{message.content}</p>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Input Bar */}
      <div className="relative glass-effect rounded-2xl p-2 border border-primary/20 cosmic-shadow">
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleVoiceInput}
            className={cn(
              "rounded-full w-10 h-10 transition-colors",
              isListening && "bg-primary text-primary-foreground animate-pulse"
            )}
          >
            <Mic className="h-5 w-5" />
          </Button>

          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message..."
            className="flex-1 bg-transparent border-none focus-visible:ring-0 text-foreground placeholder:text-muted-foreground"
          />

          <Button
            onClick={handleSend}
            size="icon"
            className="rounded-full w-10 h-10 bg-primary hover:bg-primary/90 glow-effect"
          >
            <Send className="h-5 w-5" />
          </Button>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
