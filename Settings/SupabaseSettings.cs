namespace PerceptAI.API.Settings
{
    /// <summary>
    /// Configurações do cliente Supabase, carregadas do appsettings.json
    /// ou de variáveis de ambiente via SupabaseSettings__Url / SupabaseSettings__ServiceRoleKey.
    /// </summary>
    public class SupabaseSettings
    {
        public const string SectionName = "SupabaseSettings";

        /// <summary>URL do projeto Supabase (ex: https://xyz.supabase.co).</summary>
        public string Url { get; set; } = string.Empty;

        /// <summary>
        /// Service Role Key — permite operações bypass de RLS no servidor.
        /// NUNCA exponha esta chave no frontend.
        /// </summary>
        public string ServiceRoleKey { get; set; } = string.Empty;
    }
}
