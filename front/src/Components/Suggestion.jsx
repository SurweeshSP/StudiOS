function Suggestion() {
  return (
    <div className="h-screen  flex justify-center items-center bg-white">
      <div className="p-8 bg-[#F5F4E9] rounded-2xl shadow-lg text-center w-full max-w-md">
        <h1 className="text-2xl font-bold mb-4">Suggestion Page</h1>
        <p className="text-gray-600 mb-6">
          Share your suggestions to improve EduLoop. We value your feedback.
        </p>

        <form className="space-y-4">
          <div>
            <label className="block text-gray-700 mb-1 text-left">Your Name</label>
            <input
              type="text"
              placeholder="Enter your name"
              className="w-full px-4 bg-white py-2 border rounded-lg focus:outline-none focus:ring focus:ring-blue-300"
            />
          </div>

          <div>
            <label className="block text-gray-700 mb-1 text-left">Suggestion</label>
            <textarea
              placeholder="Write your suggestion here..."
              rows="4"
              className="w-full px-4 py-2 border bg-white rounded-lg focus:outline-none focus:ring focus:ring-blue-300"
            />
          </div>

          <button
            type="submit"
            className="w-full px-6 py-2 bg-[#E4E4E4] text-black rounded-lg hover:bg-[#D4D4D4] transition"
          >
            Submit
          </button>
        </form>
      </div>
    </div>
  );
}

export default Suggestion;
