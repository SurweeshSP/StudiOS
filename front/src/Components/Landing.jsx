import Navbar from "./Navbar";
import ScrollStack, { ScrollStackItem } from "./ScrollStack";
import Footer from "./Footer";
import PixelBlast from './PixelBlast';

export default function LandingPage() {
  return (
    <div className="w-full h-screen bg-gray-50 text-gray-900">
      <Navbar />

<section className="flex flex-col items-center justify-center h-[80vh] px-8 text-center">

    
        <h1 className="text-5xl font-bold mb-4">Welcome to EduLoop</h1>
        <p className="text-lg text-gray-600 max-w-2xl">
          A modern learning platform that blends AI, collaboration, and
          personalized learning to help students and teachers thrive.
        </p>
        <button className="mt-8 px-6 py-3 bg-indigo-600 text-white rounded-full hover:bg-indigo-700 transition">Get Started</button>
      </section>

      {/* Features with ScrollStack */}
      <ScrollStack
        itemDistance={120}
        itemScale={0.05}
        itemStackDistance={40}
        blurAmount={2}    // 👈 kept blur for depth
        rotationAmount={0} // 👈 no rotation
        onStackComplete={() => console.log("Feature stack animation done!")}
      >
        <ScrollStackItem itemClassName="bg-white">
          <h2 className="text-2xl font-semibold mb-3">📊 Smart Dashboard</h2>
          <p className="text-gray-600">
            Get a unified view of your courses, progress, and analytics all in
            one place.
          </p>
        </ScrollStackItem>

        <ScrollStackItem itemClassName="bg-indigo-50">
          <h2 className="text-2xl font-semibold mb-3">💬 AI-Powered Chat</h2>
          <p className="text-gray-600">
            Learn faster with an intelligent assistant that answers your
            questions instantly.
          </p>
        </ScrollStackItem>

        <ScrollStackItem itemClassName="bg-green-50">
          <h2 className="text-2xl font-semibold mb-3">🤝 Collaboration</h2>
          <p className="text-gray-600">
            Connect with peers and mentors through real-time discussions and
            project work.
          </p>
        </ScrollStackItem>

        <ScrollStackItem itemClassName="bg-yellow-50">
          <h2 className="text-2xl font-semibold mb-3">✨ Personalized Suggestions</h2>
          <p className="text-gray-600">
            Receive tailored content, study plans, and resource
            recommendations just for you.
          </p>
        </ScrollStackItem>
      </ScrollStack>
      
      <Footer/>
    </div>
  );
}
