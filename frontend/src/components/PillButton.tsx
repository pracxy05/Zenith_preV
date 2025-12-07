import { cn } from "@/lib/utils";
import { Button } from "./ui/button";

interface PillButtonProps {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: "primary" | "secondary";
  className?: string;
}

const PillButton = ({
  children,
  onClick,
  variant = "primary",
  className,
}: PillButtonProps) => {
  return (
    <Button
      onClick={onClick}
      className={cn(
        "rounded-full px-8 py-3 font-semibold tracking-wide transition-all",
        "glass-effect border border-primary/30",
        "hover:scale-105 hover:glow-effect hover:border-primary",
        variant === "primary" && "bg-primary text-primary-foreground",
        variant === "secondary" && "bg-secondary text-secondary-foreground",
        className
      )}
    >
      {children}
    </Button>
  );
};

export default PillButton;
