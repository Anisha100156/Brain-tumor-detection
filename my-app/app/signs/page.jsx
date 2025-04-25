"use client";

import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Spotlight } from "@/components/ui/Spotlight";
import { TextGenerateEffect } from "@/components/ui/text-generate";
import Waves from "@/components/ui/Waves";
import Navigation from "@/components/ui/navigation";
import HelpSection from "@/components/ui/help";
import Footer from "@/components/ui/contactus";
import Wavesx from "@/components/ui/Wavingx";
import BrainTumorSymptoms from "@/components/ui/signsand";
import SocialWorkServices from "@/components/ui/symptoms";

const words = "Every tumor tells a different story through the symptoms it causes";

export default async  function BrainTumorNetwork() {
 
  const router = useRouter();
  

  return (
    <>
      {/* Main Hero Section */}
      <div className="relative h-180 w-full bg-cover bg-center bg-[url(/signand.jpg)]">
        
        {/* Black Overlay */}
        <div className="absolute inset-0 bg-black opacity-60 z-0"></div>

        {/* Navbar */}
        <nav className="relative z-10 flex items-center justify-between p-12 bg-[#3B4461]">
          <div className="text-2xl font-bold flex items-center ml-30">
            <img
              src="https://www.braintumornetwork.org/wp-content/themes/elevation_theme/assets/images/logo.svg"
              alt="Brain Tumor Network"
              className="h-15"
            />
          </div>
          <div className="hidden md:flex items-center gap-4 mr-30">
            <Button
              className="bg-[#FFFF] text-[#164F90] px-8 py-7 rounded-full hover:bg-gray-400 text-lg"
              onClick={() => router.push("/tumor-detection")}
            >
              Login
            </Button>
            <Button className="bg-[#164F90] text-[#FFFF] px-8 py-7 rounded-full hover:bg-[#4F67A4] text-lg" onClick={() => router.push("/stats")}>
              Our Impact
            </Button>
            <Button className="bg-purple-400 text-white px-8 py-7 rounded-full hover:bg-purple-500 text-lg" onClick={() => router.push("/signs")}>
              Signs
            </Button>
          </div>
        </nav>

        {/* Main Content */}
        <div className="relative z-10 text-left mt-20 px-5">
          <TextGenerateEffect words={words} />
        </div>

        {/* Waves at the bottom */}
        <div className="absolute bottom-0 left-0 w-full z-10">
          <Wavesx/>
        </div>
      </div>
      

     <div  className="relative z-10 -mt-18">
      <BrainTumorSymptoms/>
     </div>
    

      {/* Footer */}
      <div>
        <Footer />
      </div>
    </>
  );
}
