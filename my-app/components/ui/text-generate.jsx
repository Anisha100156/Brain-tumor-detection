"use client";
import { useEffect } from "react";
import { motion, stagger, useAnimate } from "motion/react";
import { cn } from "@/lib/utils";

export const TextGenerateEffect = ({
  words,
  className,
  filter = true,
  duration = 0.5
}) => {
  const [scope, animate] = useAnimate();
  let wordsArray = words.split(" ");

  useEffect(() => {
    animate("span", {
      opacity: 1,
      filter: filter ? "blur(0px)" : "none",
    }, {
      duration: duration || 1,
      delay: stagger(0.2),
    });
  }, [scope.current]);

  const renderWords = () => {
    return (
      <motion.div ref={scope}>
        {wordsArray.map((word, idx) => (
          <motion.span
            key={word + idx}
            className="dark:text-[#F9F7F1] text-[#F9F7F1] opacity-0"
            style={{
              fontFamily: "Inter, sans-serif",
              fontWeight: "700",  // 🔹 Bolder font
              fontSize: "84px",
              lineHeight: "82.3px",
              letterSpacing: "-3.0px",
              filter: filter ? "blur(10px)" : "none",
            }}>
            {word}{" "}
          </motion.span>
        ))}
      </motion.div>
    );
  };

  return (
    <div className={cn("font-inter", className)}>
      <div className="mt-[5px] ml-40 mr-10"> {/* 🔹 Added margin-left */}
        <div className="dark:text-[#F9F7F1] text-[#F9F7F1]">
          {renderWords()}
        </div>
      </div>
    </div>
  );
};
