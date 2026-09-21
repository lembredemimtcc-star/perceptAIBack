using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using PerceptAI.API.Services;

namespace PerceptAI.API.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class SupabaseController : ControllerBase
    {
        private readonly SupabaseService _supabaseService;

        public SupabaseController(SupabaseService supabaseService)
        {
            _supabaseService = supabaseService;
        }

        [HttpGet("test")] // GET /api/supabase/test
        public async Task<IActionResult> Test()
        {
            var json = await _supabaseService.FetchDetectionsRawAsync(1);
            return Content(json, "application/json");
        }

        [HttpPost("insert-test")] // POST /api/supabase/insert-test
        public async Task<IActionResult> InsertTest()
        {
            var ok = await _supabaseService.SaveDetectionAsync("test-patient", "neutro", 0.5f, System.DateTime.UtcNow);
            if (!ok) return StatusCode(500, new { success = false });
            return Ok(new { success = true });
        }
    }
}
