import { cn } from "@/lib/utils";

interface AnimatedCircleProps {
  size?: "large" | "medium";
  isListening?: boolean;
  isSpeaking?: boolean;
  onClick?: () => void;
}

const AnimatedCircle = ({
  size = "large",
  isListening = false,
  isSpeaking = false,
  onClick,
}: AnimatedCircleProps) => {
  const sizeClasses = {
    large: "w-[400px] h-[400px]",
    medium: "w-[320px] h-[320px]",
  };

  return (
    <div
      onClick={onClick}
      className={cn(
        "relative rounded-full cursor-pointer transition-all duration-500",
        "hover:scale-105 glow-effect",
        sizeClasses[size],
        onClick && "hover:shadow-2xl"
      )}
      style={{
        background: "var(--gradient-circle)",
      }}
    >
      {/* Outer rotating ring */}
      <div
        className="absolute inset-0 rounded-full border-4 border-accent/30 animate-cosmic-rotate"
        style={{
          borderStyle: "dashed",
        }}
      />

      {/* Middle pulsing ring */}
      <div
        className={cn(
          "absolute inset-4 rounded-full border-2 border-primary/40",
          isListening || isSpeaking ? "animate-cosmic-pulse" : ""
        )}
      />

      {/* Inner core */}
      <div className="absolute inset-8 rounded-full bg-gradient-to-br from-accent/80 via-primary/60 to-secondary/80 animate-cosmic-pulse" />

      {/* Center glow */}
      <div className="absolute inset-1/3 rounded-full bg-accent/60 blur-xl" />

      {/* Ripple effect when speaking */}
      {isSpeaking && (
        <>
          <div className="absolute inset-0 rounded-full border-4 border-primary/30 animate-ping" />
          <div className="absolute inset-0 rounded-full border-4 border-secondary/30 animate-ping animation-delay-200" />
        </>
      )}

      {/* Listening indicator */}
      {isListening && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="flex space-x-1">
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className="w-1 h-8 bg-foreground rounded-full animate-pulse"
                style={{
                  animationDelay: `${i * 0.15}s`,
                }}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default AnimatedCircle;
