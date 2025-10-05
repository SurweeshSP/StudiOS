function Footer() {
  return (
    <footer className="bg-black text-white py-6 mt-10">
      <div className="mx-auto flex flex-col md:flex-row items-center justify-between px-5">
        <h1 className="text-xl font-bold text-white">EduLoop</h1>
        <ul className="flex  space-x-6 mt-4 md:mt-0">
          <li><a href="#" className="hover:text-gray-300 transition">Home</a></li>
          <li><a href="#" className="hover:text-gray-300 transition">Dashboard</a></li>
          <li><a href="#" className="hover:text-gray-300 transition">Chat</a></li>
          <li><a href="#" className="hover:text-gray-300 transition">Suggestion</a></li>
        </ul>
        <p className="mt-4 md:mt-0 text-sm text-gray-400">
          © {new Date().getFullYear()} EduLoop. All rights reserved.
        </p>
      </div>
    </footer>
  )
}

export default Footer
