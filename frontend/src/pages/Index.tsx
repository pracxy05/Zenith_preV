import { useState } from "react";
import { cn } from "@/lib/utils";
import CosmicBackground from "@/components/CosmicBackground";
import AnimatedCircle from "@/components/AnimatedCircle";
import ThemeToggle from "@/components/ThemeToggle";
import ChatInterface from "@/components/ChatInterface";
import PillButton from "@/components/PillButton";
import { History, Settings, Sparkles } from "lucide-react";

type AppState = "idle" | "active";

const Index = () => {
  const [appState, setAppState] = useState<AppState>("idle");

  const handleCircleClick = () => {
    setAppState("active");
  };

  const handleHomeClick = () => {
    setAppState("idle");
  };

  return (
    <div className="relative min-h-screen overflow-hidden">
      <CosmicBackground />
      <ThemeToggle />

      {/* Connection Status Indicator */}
      <div className="fixed bottom-6 right-6 flex items-center gap-2 z-40">
        <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
        <span className="text-xs text-muted-foreground">Connected</span>
      </div>

      {/* Version Number */}
      <div className="fixed bottom-6 left-6 text-xs text-muted-foreground/40 z-40">
        v1.0.0
      </div>

      {/* STATE 1: IDLE/WELCOME SCREEN */}
      <div
        className={cn(
          "fixed inset-0 flex items-center justify-center transition-cosmic",
          appState === "active" && "scale-[0.6] opacity-0 pointer-events-none"
        )}
      >
        <AnimatedCircle size="large" onClick={handleCircleClick} />
      </div>

      {/* STATE 2: ACTIVE INTERFACE */}
      <div
        className={cn(
          "transition-cosmic",
          appState === "idle" && "opacity-0 pointer-events-none scale-95"
        )}
      >
        {/* Header */}
        <header className="fixed top-6 left-6 z-40 flex gap-3 animate-fade-in">
          <PillButton onClick={handleHomeClick}>HOME</PillButton>
          <PillButton variant="secondary">
            <History className="w-4 h-4 mr-2" />
            HISTORY
          </PillButton>
        </header>

        {/* Main Content */}
        <main className="flex flex-col items-center justify-center min-h-screen pt-20 pb-32">
          {/* ZENITH Title */}
          <h1
            className="text-8xl font-bold mb-12 tracking-[0.2em] text-foreground animate-fade-in"
            style={{
              textShadow: "0 0 40px hsla(var(--primary), 0.5)",
            }}
          >
            ZENITH
          </h1>

          {/* Central Circle */}
          <div className="mb-16 animate-scale-in">
            <AnimatedCircle size="medium" />
          </div>

          {/* Chat Interface */}
          <div className="w-full animate-fade-in" style={{ animationDelay: "0.2s" }}>
            <ChatInterface />
          </div>
        </main>

        {/* Footer */}
        <footer className="fixed bottom-6 left-1/2 -translate-x-1/2 z-40 flex gap-3 animate-fade-in">
          <PillButton variant="secondary">
            <Settings className="w-4 h-4 mr-2" />
            SETTINGS
          </PillButton>
          <PillButton>
            <Sparkles className="w-4 h-4 mr-2" />
            AI
          </PillButton>
        </footer>
      </div>
    </div>
  );
};

export default Index;
