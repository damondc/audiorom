// Headless libretro runner: render a ROM's audio to a WAV with any libretro core.
// usage: runcore core.so rom seconds out.wav
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>
struct gi { const char *path; const void *data; size_t size; const char *meta; };
struct avi { struct { unsigned bw, bh, mw, mh; float ar; } g; struct { double fps, sr; } t; };
static int16_t *buf; static size_t n, cap;
static void push(int16_t l, int16_t r){ if(n+2>cap){cap=cap?cap*2:1<<20;buf=realloc(buf,cap*2);} buf[n++]=l; buf[n++]=r; }
static void as(int16_t l,int16_t r){ push(l,r); }
static size_t asb(const int16_t *d,size_t f){ for(size_t i=0;i<f;i++) push(d[2*i],d[2*i+1]); return f; }
static int pixfmt=2; static uint8_t *fb; static unsigned fw,fh,fp;
static void vr(const void*d,unsigned w,unsigned h,size_t p){ if(!d) return; fb=realloc(fb,p*h); memcpy(fb,d,p*h); fw=w;fh=h;fp=p; }
static void dump(const char*path){ if(!fb) return; FILE*o=fopen(path,"wb"); fprintf(o,"P6\n%u %u\n255\n",fw,fh);
  for(unsigned y=0;y<fh;y++) for(unsigned x=0;x<fw;x++){ uint8_t r,g,b;
    if(pixfmt==2){ uint16_t v=*(uint16_t*)(fb+y*fp+x*2); r=((v>>11)&31)*255/31; g=((v>>5)&63)*255/63; b=(v&31)*255/31; }
    else { uint32_t v=*(uint32_t*)(fb+y*fp+x*4); r=v>>16; g=v>>8; b=v; }
    fputc(r,o);fputc(g,o);fputc(b,o);} fclose(o);}
static void ip(void){}
static int cur_frame; static int nev; static int ev_f[64], ev_id[64], ev_len[64];
static int16_t is(unsigned a,unsigned b,unsigned c,unsigned d){
  if(a!=0||b!=1) return 0;   // port 0, RETRO_DEVICE_JOYPAD
  for(int i=0;i<nev;i++) if((int)d==ev_id[i] && cur_frame>=ev_f[i] && cur_frame<ev_f[i]+ev_len[i]) return 1;
  return 0; }
static int env(unsigned cmd, void *data){
  switch(cmd){
    case 10: pixfmt=*(int*)data; return 1;                       // SET_PIXEL_FORMAT
    case 9: case 31: *(const char**)data="."; return 1;  // system/save dir
    default: return 0;
  }
}
int main(int argc,char**argv){
  void *h=dlopen(argv[1],RTLD_NOW); if(!h){fprintf(stderr,"%s\n",dlerror());return 1;}
  void (*set_env)(void*)=dlsym(h,"retro_set_environment");
  void (*set_vr)(void*)=dlsym(h,"retro_set_video_refresh");
  void (*set_as)(void*)=dlsym(h,"retro_set_audio_sample");
  void (*set_asb)(void*)=dlsym(h,"retro_set_audio_sample_batch");
  void (*set_ip)(void*)=dlsym(h,"retro_set_input_poll");
  void (*set_is)(void*)=dlsym(h,"retro_set_input_state");
  void (*init)(void)=dlsym(h,"retro_init");
  int (*load)(const struct gi*)=dlsym(h,"retro_load_game");
  void (*avinfo)(struct avi*)=dlsym(h,"retro_get_system_av_info");
  void (*run)(void)=dlsym(h,"retro_run");
  set_env(env); init(); set_vr(vr); set_as(as); set_asb(asb); set_ip(ip); set_is(is);
  FILE*f=fopen(argv[2],"rb"); fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
  void*rom=malloc(sz); fread(rom,1,sz,f); fclose(f);
  struct gi g={argv[2],rom,(size_t)sz,NULL};
  if(!load(&g)){fprintf(stderr,"load failed\n");return 1;}
  struct avi av; avinfo(&av);
  double secs=atof(argv[3]); int frames=(int)(secs*av.t.fps);
  fprintf(stderr,"fps %.3f sample rate %.0f\n",av.t.fps,av.t.sr);
  for(int i=6;i<argc;i++){ int f,id,l; if(sscanf(argv[i],"%d:%d:%d",&f,&id,&l)==3){ ev_f[nev]=f;ev_id[nev]=id;ev_len[nev]=l;nev++; } }
  for(int i=0;i<frames;i++){ cur_frame=i; run(); }
  FILE*o=fopen(argv[4],"wb"); uint32_t sr=(uint32_t)av.t.sr, db=n*2, v;
  fwrite("RIFF",1,4,o); v=36+db; fwrite(&v,4,1,o); fwrite("WAVEfmt ",1,8,o); v=16; fwrite(&v,4,1,o);
  uint16_t s=1; fwrite(&s,2,1,o); s=2; fwrite(&s,2,1,o); fwrite(&sr,4,1,o); v=sr*4; fwrite(&v,4,1,o);
  s=4; fwrite(&s,2,1,o); s=16; fwrite(&s,2,1,o); fwrite("data",1,4,o); fwrite(&db,4,1,o);
  fwrite(buf,2,n,o); fclose(o); fprintf(stderr,"wrote %zu frames\n",n/2); if(argc>5) dump(argv[5]); return 0;
}
